from __future__ import annotations

import json
import re
import os
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from omegaconf import OmegaConf
from tqdm import tqdm
from openai import OpenAI
import hydra
import time
from dotenv import load_dotenv

# TTS imports
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import resample
from audiostretchy.stretch import AudioStretch

# Load environment variables from .env file
load_dotenv()


APIKEY = os.getenv("OPENROUTER_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════
# TTS INITIALIZATION - IndexTTS2
# ═══════════════════════════════════════════════════════════════════════════

# Import IndexTTS2 model
from indextts.infer_v2 import IndexTTS2

# ═══════════════════════════════════════════════════════════════════════════
# NOTE: Paralinguistic pools and control templates are now in config file
# See conf/base_para.yaml for configuration
# ═══════════════════════════════════════════════════════════════════════════

# ──────────────────────────────  NORMAL UTILS  ────────────────────────────────

def convert_nested_json_to_jsonl(input_path, output_path):
    """Convert nested JSON to JSONL format"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError("No 'scenarios' key or empty list in the input JSON.")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in scenarios:
            scenario_id, inner = next(iter(entry.items()))
            out_obj = {
                "id": scenario_id,
                "description": inner["description"]
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Converted {len(scenarios)} scenarios to JSONL → {output_path}")

# ──────────────────────────────  PARALINGUISTIC UTILS  ────────────────────────────────

def generate_single_paralinguistic_tag(pools: Dict[str, List[str]], dimension: str = None) -> Dict[str, str]:
    """
    Select ONLY ONE paralinguistic dimension and value (not all dimensions).

    Args:
        pools: Dictionary of paralinguistic dimension pools from config
        dimension: Optional specific dimension to use. If None, randomly select one.

    Returns: Dictionary with SINGLE dimension and value
             e.g., {"speed": "fast"} or {"emotion": "happy"}
    """
    if dimension is None:
        # Randomly pick ONE dimension
        dimension = random.choice(list(pools.keys()))

    value = random.choice(pools[dimension])
    return {dimension: value}

def format_paralinguistic_tags(tags: Dict[str, str]) -> str:
    """
    Format paralinguistic tags into parentheses format (like old version).
    Format: (speed:fast) or (emotion:happy) - SINGLE dimension only!
    """
    tag_str = ", ".join([f"{k}:{v}" for k, v in tags.items()])
    return f"({tag_str})"

def extract_paralinguistic_tags(text: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Extract paralinguistic tags from parentheses format (like old version TTS extraction).
    Format: (gender:woman, age:young_adult, ...) utterance text
    Returns: (clean_text, tags_dict)
    """
    # Check if text starts with parentheses
    if text.startswith("("):
        closing = text.find(")")
        if closing != -1:
            tag_string = text[1:closing].strip()  # Extract content inside ()
            clean_text = text[closing+1:].strip()  # Get text after )

            # Parse the tags (format: key:value, key:value)
            tags = {}
            for pair in tag_string.split(','):
                if ':' in pair:
                    key, value = pair.strip().split(':', 1)
                    tags[key.strip()] = value.strip()

            return clean_text, tags

    return text, None

def save_dialogue(output_dir: Path, dialogue_id: str, dialogue_turns: List[str]):
    """
    Save dialogue as simple .txt file (same as old version).
    Tags are embedded in parentheses format: (gender:woman, speed:fast, ...) utterance
    TTS will extract tags when needed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dialogue_path = output_dir / f"{dialogue_id}.txt"
    dialogue_path.write_text("\n".join(dialogue_turns), encoding="utf-8")

# ──────────────────────────────  LLM UTILS  ────────────────────────────────

def chat_completion(model_name: str, messages: List[Dict], **gen_kwargs) -> str:
    """
    Unified chat completion function that works with OpenRouter API.
    """
    api_key = APIKEY
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Please set it in your .env file")

    base_url = "https://openrouter.ai/api/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        **gen_kwargs,
    )
    return completion.choices[0].message.content.strip()

# ─────────────────────────────  SCENARIO GEN  ──────────────────────────────

def generate_scenarios(cfg):
    """Generate scenarios for dialogues"""
    out_path = Path(cfg.scenario["out_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = cfg.scenario["prompt"].format(n=cfg.scenario["n"])
    msgs = [{"role": "system", "content": system_prompt}]

    text = chat_completion(
        cfg.scenario["model"],
        msgs,
    )

    print(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text_clean = re.sub(r"```(?:json)?|```", "", text, flags=re.I).strip()
        data = json.loads(text_clean)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info("Saved %s → %s", type(data).__name__, out_path)

# ─────────────────────────────  CONTROL MODE UTILS  ──────────────────────────────

def select_control_target(pools: Dict[str, List[str]], fixed_dimension: str, current_value: str = None) -> Tuple[str, str]:
    """
    Select a paralinguistic target value for the specified dimension.
    EXCLUDES the current value to ensure meaningful control requests.

    Args:
        pools: Paralinguistic dimension pools from config
        fixed_dimension: The specific dimension to use (from config)
        current_value: Current value of this dimension (will be excluded from selection)

    Returns: (dimension, target_value)
             e.g., ("speed", "fast") or ("emotion", "happy")
    """
    # Use the fixed dimension from config
    dimension = fixed_dimension
    available_values = pools[dimension]

    # Exclude current value if provided (can't request what you already have)
    if current_value and current_value in available_values:
        available_values = [v for v in available_values if v != current_value]
        logging.info(f"    [SELECT_CONTROL] Excluding current value '{current_value}' from selection")

    # If we filtered out all values, fall back to original pool (edge case)
    if not available_values:
        logging.warning(f"    [SELECT_CONTROL] All values filtered out, using full pool")
        available_values = pools[dimension]

    target_value = random.choice(available_values)
    return dimension, target_value

# ─────────────────────────────  PARALINGUISTIC DIALOGUE GEN  ──────────────────────────────

def generate_paralinguistic_dialogues(cfg):
    """
    PARALINGUISTIC VERSION: Generates dialogues with paralinguistic control tags.

    MODES (randomly switch within each dialogue):
    - USUAL MODE: Normal conversation with automatic paralinguistic tags
    - CONTROL MODE: User explicitly requests voice characteristic changes

    Key features:
    1. Randomly generates paralinguistic attributes for agent utterances
    2. Includes tags in the LLM prompt to guide generation
    3. Agent responses include paralinguistic tags in parentheses format: (tags)
    4. Saves dialogue as simple .txt file (TTS extracts tags later)
    5. Random switching: Each user turn has control_request_frequency chance to make control request
    """
    # Read scenarios
    scen_path = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    out_dir = Path(cfg.dialogue["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [json.loads(l) for l in scen_path.read_text(encoding="utf-8").splitlines()]

    # Extract configuration
    user_model = cfg.dialogue["user_model"]
    agent_model = cfg.dialogue["agent_model"]
    user_system_prompt = cfg.dialogue["user_prompt"]
    agent_system_prompt = cfg.dialogue["agent_prompt"]
    control_user_prompt = cfg.dialogue.get("control_user_prompt", user_system_prompt)
    control_agent_prompt = cfg.dialogue.get("control_agent_prompt", agent_system_prompt)
    max_turns = cfg.dialogue.get("max_turns", 10)
    min_turns = cfg.dialogue.get("min_turns", 4)
    dialogues_per_scenario = cfg.dialogue.get("per_scenario", 3)

    # Control mode configuration
    # Two-tier probability system:
    # - First control request: Higher probability to ensure most dialogues have at least one control
    # - Subsequent requests: Lower probability to avoid overwhelming the dialogue with controls
    control_request_frequency_first = cfg.dialogue.get("control_request_frequency_first", 0.8)  # 80% for first control
    control_request_frequency_subsequent = cfg.dialogue.get("control_request_frequency_subsequent", 0.4)  # 40% for subsequent
    control_dimension = cfg.dialogue.get("control_dimension", "speed")  # Fixed dimension from config

    # Load paralinguistic pools from config (single source of truth)
    paralinguistic_pools = cfg.get("paralinguistic_pools", {})

    logging.info(f"Generating PARALINGUISTIC dialogues:")
    logging.info(f"  User Model: {user_model}")
    logging.info(f"  Agent Model: {agent_model}")
    logging.info(f"  Turn Range: {min_turns}-{max_turns}")
    logging.info(f"  Dialogues per Scenario: {dialogues_per_scenario}")
    logging.info(f"  Control Request Frequency (First): {control_request_frequency_first}")
    logging.info(f"  Control Request Frequency (Subsequent): {control_request_frequency_subsequent}")
    logging.info(f"  Control Dimension (Fixed): {control_dimension}")

    for scenario in tqdm(scenarios, desc="dialogues"):
        scenario_desc = json.dumps(scenario, ensure_ascii=False)

        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_turns = []
            conversation_history = []
            control_request_info = None  # Initialize control signal (passed from User to Agent)

            # Track if this dialogue has had any control request yet (for two-tier probability)
            has_had_control_request = False

            # Track last emotion tag for each role (for continuity in normal mode)
            last_user_emotion_tag = None  # Format: (dimension, value) e.g., ("emotion", "happy")
            last_agent_emotion_tag = None

            context_info = f"Scenario: {scenario_desc}\n\n"
            num_turns = random.randint(min_turns, max_turns)
            # Ensure num_turns is even (dialogues must end with agent response)
            if num_turns % 2 != 0:
                num_turns += 1
            logging.info(f"  Generating {scenario['id']}_{dialogue_idx + 1} with {num_turns} turns")

            for turn in range(num_turns):
                if turn % 2 == 0:
                    # ============ USER's TURN ============
                    # RANDOM SWITCH: Decide if this user turn should include a control request
                    # Only allow control requests after turn 0 (first turn should be normal greeting)
                    # Control signal will be passed to the next agent turn
                    # TWO-TIER PROBABILITY SYSTEM:
                    # - First control request: Higher probability (e.g., 0.8) to ensure coverage
                    # - Subsequent requests: Lower probability (e.g., 0.4) to avoid overwhelming dialogue
                    can_make_control_request = turn > 0

                    if can_make_control_request:
                        if has_had_control_request:
                            # Already had a control request - use LOWER probability for subsequent requests
                            is_control_request = random.random() < control_request_frequency_subsequent
                            if is_control_request:
                                logging.info(f"    [CONTROL DECISION] Turn {turn}: Subsequent control (prob={control_request_frequency_subsequent})")
                        else:
                            # First control request - use HIGHER probability
                            is_control_request = random.random() < control_request_frequency_first
                            if is_control_request:
                                logging.info(f"    [CONTROL DECISION] Turn {turn}: First control (prob={control_request_frequency_first})")
                    else:
                        is_control_request = False

                    # Initialize control_request_info for this user turn
                    # This will be used by the agent in the NEXT turn (turn + 1)
                    current_turn_control_info = None

                    if is_control_request:
                        # Mark that this dialogue has now had a control request
                        if not has_had_control_request:
                            has_had_control_request = True
                            logging.info(f"    [CONTROL TRACKING] First control request marked for this dialogue")
                        # Determine current value to exclude from selection
                        # Check if last agent tag matches the control dimension
                        current_value_to_exclude = None
                        if last_agent_emotion_tag:
                            last_dim, last_val = last_agent_emotion_tag
                            if last_dim == control_dimension:
                                current_value_to_exclude = last_val
                                logging.info(f"    [CONTROL] Current agent state: {last_dim}={last_val}")

                        # Select a control target value for the fixed dimension (excluding current value)
                        ctrl_dim, ctrl_value = select_control_target(
                            paralinguistic_pools, control_dimension, current_value_to_exclude
                        )
                        current_turn_control_info = (ctrl_dim, ctrl_value)
                        logging.info(f"    [CONTROL] Turn {turn}: User requesting {ctrl_dim}={ctrl_value}")

                    if turn == 0:
                        # First turn - start normal conversation
                        user_instruction = (
                            f"{context_info}"
                            f"This is the start of the conversation. "
                            f"Generate the user's first message based on the scenario. "
                            f"Be natural and conversational. Only output the user's utterance, nothing else."
                        )
                    else:
                        history_text = "\n".join(conversation_history)

                        if is_control_request:
                            # CONTROL MODE: User requests voice change (LLM generates diverse phrasing)
                            user_instruction = (
                                f"{context_info}"
                                f"Conversation so far:\n{history_text}\n\n"
                                f"Generate the user's next response where they ask the agent to change their voice.\n"
                                f"Specifically, ask them to adjust their {ctrl_dim} to be '{ctrl_value}'.\n"
                                f"Be creative and natural - use your own words, don't use a template.\n"
                                f"Examples of natural requests:\n"
                                f"- For speed=fast: 'Can you talk faster?', 'Please speed up', 'Speak quicker'\n"
                                f"- For emotion=happy: 'Sound happier!', 'Be more cheerful', 'Can you be more upbeat?'\n"
                                f"- For volume=loud: 'Speak louder', 'I can't hear you well', 'Increase your volume'\n\n"
                                f"Generate a DIVERSE, NATURAL request for {ctrl_dim}='{ctrl_value}'.\n"
                                f"Then optionally continue with the conversation topic.\n"
                                f"Only output the user's utterance, nothing else."
                            )
                        else:
                            # USUAL MODE: Normal conversation turn
                            user_instruction = (
                                f"{context_info}"
                                f"Conversation so far:\n{history_text}\n\n"
                                f"Generate the user's next response based on the agent's last message. "
                                f"Be natural and conversational. Only output the user's utterance, nothing else."
                            )

                    user_messages = [
                        {"role": "system", "content": control_user_prompt},
                        {"role": "user", "content": user_instruction}
                    ]

                    # Retry logic
                    max_retries = 3
                    user_utterance = None

                    for retry in range(max_retries):
                        try:
                            user_utterance = chat_completion(
                                user_model,
                                user_messages,
                                max_tokens=512,
                                temperature=0.9,
                                top_p=0.9
                            ).strip()

                            user_utterance = re.sub(r'^User:\s*', '', user_utterance, flags=re.IGNORECASE)

                            if user_utterance:
                                break
                            else:
                                logging.warning(f"Empty response from user LLM at turn {turn}, retry {retry + 1}/{max_retries}")

                        except Exception as e:
                            logging.error(f"Error in user LLM turn {turn} (retry {retry + 1}/{max_retries}): {e}")
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry
                                logging.info(f"Waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                            else:
                                logging.error(f"Failed after {max_retries} retries, breaking dialogue...")

                    if user_utterance:
                        dialogue_turns.append(f"User: {user_utterance}")
                        conversation_history.append(f"User: {user_utterance}")
                        # Pass control signal to next agent turn
                        control_request_info = current_turn_control_info
                    else:
                        logging.error(f"Could not generate user response after {max_retries} retries, ending dialogue early")
                        break

                else:
                    # ============ AGENT's TURN ============

                    # Check if user made a control request in previous turn
                    if control_request_info:
                        # CONTROL MODE: User requested voice change - Agent applies it
                        ctrl_dim, ctrl_value = control_request_info

                        # Create SINGLE tag for requested dimension
                        para_tags = {ctrl_dim: ctrl_value}
                        para_tag_str = format_paralinguistic_tags(para_tags)

                        # Instruction for agent to acknowledge the control request
                        para_instruction = (
                            f"\n\nIMPORTANT: The user just asked you to change your {ctrl_dim} to '{ctrl_value}'.\n"
                            f"You should:\n"
                            f"1. Acknowledge the request naturally (e.g., 'Sure!', 'Of course', 'No problem', 'Okay!')\n"
                            f"2. Then continue helping with their question/topic\n\n"
                            f"Your response MUST start with the tag in parentheses EXACTLY as shown:\n"
                            f"{para_tag_str}\n\n"
                            f"Format: {para_tag_str} [brief acknowledgment] [continue conversation]\n"
                        )

                        # Update last emotion tag for agent
                        last_agent_emotion_tag = (ctrl_dim, ctrl_value)

                        # Reset control request after processing
                        control_request_info = None

                    else:
                        # USUAL MODE - Normal agent response
                        # Check if there's a last emotion tag to maintain continuity
                        if last_agent_emotion_tag:
                            # EMOTION TAG CONTINUITY: Copy the last emotion tag
                            ctrl_dim, ctrl_value = last_agent_emotion_tag
                            para_tags = {ctrl_dim: ctrl_value}
                            para_tag_str = format_paralinguistic_tags(para_tags)

                            # Instruction to copy the emotion tag for continuity
                            para_instruction = (
                                f"\n\nIMPORTANT: For emotional continuity, maintain the same {ctrl_dim} as your last response ('{ctrl_value}').\n"
                                f"Your response MUST start with the tag in parentheses EXACTLY as shown:\n"
                                f"{para_tag_str}\n\n"
                                f"Format: {para_tag_str} [your response]\n"
                                f"This ensures your voice characteristics remain consistent throughout the conversation.\n"
                            )
                            logging.info(f"    [CONTINUITY] Turn {turn}: Agent maintaining {ctrl_dim}={ctrl_value}")
                        else:
                            # No emotion tag - normal response without tags
                            para_tags = None
                            para_instruction = ""

                    history_text = "\n".join(conversation_history)
                    agent_instruction = (
                        f"{context_info}"
                        f"Conversation so far:\n{history_text}\n\n"
                        f"Generate the agent's response to the user's last message. "
                        f"Be helpful, professional, and natural. "
                        f"{para_instruction}"
                        f"Only output the agent's utterance (with the paralinguistic tag if specified), nothing else."
                    )

                    # Choose appropriate prompt based on mode
                    if para_instruction:
                        # CONTROL MODE: Use control agent prompt with tag instructions
                        agent_messages = [
                            {"role": "system", "content": control_agent_prompt},
                            {"role": "user", "content": agent_instruction}
                        ]
                    else:
                        # USUAL MODE: Use normal agent prompt (no tags mentioned)
                        agent_messages = [
                            {"role": "system", "content": agent_system_prompt},
                            {"role": "user", "content": agent_instruction}
                        ]

                    # Retry logic
                    max_retries = 3
                    agent_utterance = None

                    for retry in range(max_retries):
                        try:
                            agent_utterance = chat_completion(
                                agent_model,
                                agent_messages,
                                max_tokens=512,
                                temperature=0.9,
                                top_p=0.9
                            ).strip()

                            # Clean up if LLM adds "Agent:" prefix
                            agent_utterance = re.sub(r'^Agent:\s*', '', agent_utterance, flags=re.IGNORECASE)

                            if agent_utterance:
                                break
                            else:
                                logging.warning(f"Empty response from agent LLM at turn {turn}, retry {retry + 1}/{max_retries}")

                        except Exception as e:
                            logging.error(f"Error in agent LLM turn {turn} (retry {retry + 1}/{max_retries}): {e}")
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry
                                logging.info(f"Waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                            else:
                                logging.error(f"Failed after {max_retries} retries, breaking dialogue...")

                    if agent_utterance:
                        # Ensure the paralinguistic tag is present if it was requested
                        if para_tags and not agent_utterance.startswith("("):
                            # If LLM didn't include the tag, add it ourselves
                            para_tag_str = format_paralinguistic_tags(para_tags)
                            agent_utterance = f"{para_tag_str} {agent_utterance}"

                        full_agent_turn = f"Agent: {agent_utterance}"
                        dialogue_turns.append(full_agent_turn)

                        # Extract tags from agent utterance and update tracking
                        clean_utterance, extracted_tags = extract_paralinguistic_tags(agent_utterance)

                        # Update last emotion tag for agent (for continuity tracking)
                        if extracted_tags:
                            # Extract the dimension and value
                            tag_dim, tag_value = next(iter(extracted_tags.items()))
                            last_agent_emotion_tag = (tag_dim, tag_value)
                            logging.info(f"    [TRACKING] Agent emotion tag updated: {tag_dim}={tag_value}")

                        # For conversation history, use clean version (without tags)
                        conversation_history.append(f"Agent: {clean_utterance}")
                    else:
                        logging.error(f"Could not generate agent response after {max_retries} retries, ending dialogue early")
                        break

            # Save dialogue (tags embedded in parentheses)
            if dialogue_turns:
                dialogue_id = f"{scenario['id']}_{dialogue_idx + 1}"
                save_dialogue(out_dir, dialogue_id, dialogue_turns)

                logging.info(f"Saved dialogue: {dialogue_id}")
                print(f"\n=== Generated Paralinguistic Dialogue {dialogue_idx + 1} for {scenario['id']} ===")
                print("\n".join(dialogue_turns[:5]))  # Print first 5 turns
                if len(dialogue_turns) > 5:
                    print("...")
                print("=" * 60)

# ────────────────────────────  POST-PROCESSING  ─────────────────────────────

def analyze_paralinguistic_distribution(cfg):
    """
    Analyze the distribution of paralinguistic attributes across all generated dialogues.
    Extracts tags from dialogue .txt files (parentheses format).
    Creates a statistical report.
    """
    src_dir = Path(cfg.dialogue["out_dir"])
    report_path = src_dir / "paralinguistic_analysis.json"

    # Load paralinguistic pools from config
    paralinguistic_pools = cfg.get("paralinguistic_pools", {})

    # Initialize counters
    distribution = {dimension: {} for dimension in paralinguistic_pools.keys()}
    total_agent_turns = 0
    turns_with_tags = 0

    # Process all dialogue .txt files
    for dialogue_file in src_dir.glob("*.txt"):
        with open(dialogue_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("Agent:"):
                total_agent_turns += 1

                # Extract tags from agent utterance
                agent_text = line.replace("Agent:", "").strip()
                clean_text, para_tags = extract_paralinguistic_tags(agent_text)

                if para_tags:
                    turns_with_tags += 1

                    # Count each dimension value
                    for dimension, value in para_tags.items():
                        if dimension in distribution:
                            if value in distribution[dimension]:
                                distribution[dimension][value] += 1
                            else:
                                distribution[dimension][value] = 1

    # Create report
    report = {
        "total_agent_turns": total_agent_turns,
        "turns_with_paralinguistic_tags": turns_with_tags,
        "coverage": turns_with_tags / total_agent_turns if total_agent_turns > 0 else 0,
        "distribution": distribution
    }

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(f"Paralinguistic analysis saved to: {report_path}")
    print(f"\n=== Paralinguistic Analysis ===")
    print(f"Total Agent Turns: {total_agent_turns}")
    print(f"Turns with Tags: {turns_with_tags}")
    print(f"Coverage: {report['coverage']:.2%}")
    print("=" * 60)

# ────────────────────────────────  TTS - XTTS2  ────────────────────────────────────

def detect_speaker_gender(spk_audio_path, spk_audio_base_dir):
    """
    Detect the gender of a speaker by checking if the audio file exists in male or female subdirectories.

    Args:
        spk_audio_path: Path to the speaker audio file
        spk_audio_base_dir: Base directory containing male/ and female/ subdirectories

    Returns:
        "male" or "female" or None if not found
    """
    spk_filename = Path(spk_audio_path).name

    # Check if file exists in male directory
    male_path = Path(spk_audio_base_dir) / "male" / spk_filename
    if male_path.exists():
        logging.info(f"  -> Detected gender: male (found in male/)")
        return "male"

    # Check if file exists in female directory
    female_path = Path(spk_audio_base_dir) / "female" / spk_filename
    if female_path.exists():
        logging.info(f"  -> Detected gender: female (found in female/)")
        return "female"

    logging.warning(f"  -> Could not detect gender for {spk_filename}")
    return None

def pre_detect_first_gender_tag(script):
    """
    Pre-scan the script to find the first gender tag in agent turns.
    This is used to determine the initial speaker gender (opposite of first switch).

    Args:
        script: List of (role, text) tuples

    Returns:
        First gender tag value (e.g., "woman", "man") or None if no gender tags found
    """
    for role, text in script:
        # Only check agent turns
        if role in ("Agent", "[overlap] Agent"):
            # Extract tags
            _, para_tags = extract_paralinguistic_tags(text)
            if para_tags and "gender" in para_tags:
                first_gender = para_tags["gender"]
                logging.info(f"  -> First gender tag detected: {first_gender}")
                return first_gender

    logging.info(f"  -> No gender tags found in script")
    return None

def get_opposite_gender(gender_value):
    """
    Get the opposite gender for initial speaker setup.

    Args:
        gender_value: Gender value from tag (e.g., "woman", "man", "female", "male")

    Returns:
        Opposite gender directory name ("male" or "female")
    """
    # Map to standard names
    gender_map = {
        "woman": "female",
        "female": "female",
        "man": "male",
        "male": "male"
    }

    target_gender = gender_map.get(gender_value.lower())

    if target_gender == "female":
        return "male"  # If switching to female, start with male
    elif target_gender == "male":
        return "female"  # If switching to male, start with female
    else:
        return None

def apply_audio_post_processing(audio_tensor, sample_rate, para_tags):
    """
    Apply post-processing to audio based on paralinguistic tags.

    Args:
        audio_tensor: torch.Tensor of shape (channels, samples)
        sample_rate: int, audio sample rate
        para_tags: dict, extracted paralinguistic tags

    Returns:
        torch.Tensor: Processed audio
    """
    if para_tags is None:
        return audio_tensor

    # Convert to numpy for processing
    audio_np = audio_tensor.cpu().numpy()

    # Apply SPEED modification (pitch-preserving time-stretch using AudioStretch)
    if "speed" in para_tags:
        speed_map = {
            "very_slow": 0.7,
            "slow": 0.85,
            "normal": 1.0,
            "fast": 1.2,
            "very_fast": 1.4
        }
        speed_factor = speed_map.get(para_tags["speed"], 1.0)

        if speed_factor != 1.0:
            logging.info(f"  -> Applying speed: {para_tags['speed']} (factor: {speed_factor})")
            logging.info(f"     Using AudioStretch (TDHS) for high-quality pitch preservation")
            logging.info(f"     Audio shape before: {audio_np.shape}, dtype: {audio_np.dtype}")

            # AudioStretch uses file-based processing, so we need temp files
            import tempfile

            # Create temporary input and output files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
                temp_input_path = temp_input.name
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_output_path = temp_output.name

            try:
                # Save current audio to temporary input file
                # Convert numpy array back to tensor for saving
                temp_audio_tensor = torch.from_numpy(audio_np).float()
                torchaudio.save(temp_input_path, temp_audio_tensor, sample_rate)

                # Convert Speed to Ratio for AudioStretch
                # Speed 2.0 (Double speed) -> Ratio 0.5 (Half length)
                # Speed 0.5 (Half speed) -> Ratio 2.0 (Double length)
                target_ratio = 1.0 / speed_factor

                logging.info(f"     Speed factor: {speed_factor}, Target ratio: {target_ratio}")

                # Apply AudioStretch with Time-Domain Harmonic Scaling (TDHS)
                stretcher = AudioStretch()
                stretcher.open(temp_input_path)

                # stretch() parameters tuned for speech
                # upper_freq/lower_freq help the pitch detector lock onto voice fundamentals
                stretcher.stretch(
                    ratio=target_ratio,
                    gap_ratio=target_ratio * 0.5,  # Speed up silence even more (smart processing)
                    upper_freq=333,  # Upper frequency for pitch detection
                    lower_freq=55    # Lower frequency for pitch detection
                )

                stretcher.save(temp_output_path)

                # Load the processed audio back
                processed_audio, processed_sr = torchaudio.load(temp_output_path)

                # Resample if necessary
                if processed_sr != sample_rate:
                    processed_audio = torchaudio.functional.resample(processed_audio, processed_sr, sample_rate)

                # Convert back to numpy
                audio_np = processed_audio.cpu().numpy()

                logging.info(f"     Audio shape after: {audio_np.shape}")

            finally:
                # Clean up temporary files
                import os
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

    # Apply PITCH modification (pitch-shift)
    # if "pitch" in para_tags:
    #     pitch_map = {
    #         "very_low": -4,     # semitones
    #         "low": -2,
    #         "normal": 0,
    #         "high": 2,
    #         "very_high": 4
    #     }
    #     pitch_shift = pitch_map.get(para_tags["pitch"], 0)

    #     if pitch_shift != 0:
    #         logging.info(f"  -> Applying pitch: {para_tags['pitch']} (shift: {pitch_shift} semitones)")
    #         # Process each channel
    #         processed_channels = []
    #         for ch in audio_np:
    #             # Pitch-shift
    #             shifted = librosa.effects.pitch_shift(ch, sr=sample_rate, n_steps=pitch_shift)
    #             processed_channels.append(shifted)
    #         audio_np = np.array(processed_channels)

    # Apply VOLUME modification (amplitude scaling)
    if "volume" in para_tags:
        volume_map = {
            "very_quiet": 0.2,
            "quiet": 0.4,
            "normal": 1.0,
            "loud": 1.8,
            "very_loud": 3.0
        }
        volume_factor = volume_map.get(para_tags["volume"], 1.0)

        if volume_factor != 1.0:
            logging.info(f"  -> Applying volume: {para_tags['volume']} (factor: {volume_factor})")
            audio_np = audio_np * volume_factor
            # Prevent clipping
            audio_np = np.clip(audio_np, -1.0, 1.0)

    # Convert back to tensor
    audio_tensor = torch.from_numpy(audio_np).float()
    return audio_tensor

def get_emotion_reference_audio(para_tags, emotion_audio_pool_dir, speaker_gender=None):
    """
    Map paralinguistic tags to emotion reference audio files.
    Randomly selects from multiple reference files in each category.
    Uses IndexTTS-2's emotion reference audio feature (separate style prompt).

    Special handling for pitch mode:
    - Uses speaker_gender to select gender-appropriate pitch reference
    - E.g., "high" pitch + male speaker → high_pitch_man
    - E.g., "low" pitch + female speaker → low_pitch_woman

    Args:
        para_tags: Dict of paralinguistic tags (e.g., {"gender": "man"} or {"age": "elderly"})
        emotion_audio_pool_dir: Base directory containing emotion reference audio pool
        speaker_gender: "male" or "female" (required for pitch mode)

    Returns:
        Path to emotion reference audio file, or None if no tags or file not found

    Audio pool structure:
        emo/
        ├── sad/
        │   ├── sad_0.mp3
        │   ├── sad_1.mp3
        │   └── ...
        ├── fast/
        │   ├── fast_0.mp3
        │   ├── fast_1.mp3
        │   └── ...
        ├── high_pitch_man/
        │   ├── high_pitch_man_0.mp3
        │   └── ...
        ├── high_pitch_woman/
        │   ├── high_pitch_woman_0.mp3
        │   └── ...
        └── ...
    """
    if not para_tags:
        return None

    # Get the dimension and value (should be only ONE per utterance)
    dimension, value = next(iter(para_tags.items()))

    logging.info(f"  -> Looking for emotion reference audio: {dimension}={value}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL HANDLING FOR PITCH MODE - GENDER-AWARE REFERENCE SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    if dimension == "pitch":
        if not speaker_gender:
            logging.warning(f"  -> ⚠️  Pitch mode requires speaker gender, but none provided. Using default.")
            return None

        # Map pitch values to gender-specific directory names
        # E.g., "high" + "male" → "high_pitch_man"
        #       "low" + "female" → "low_pitch_woman"
        pitch_value = value.replace("very_", "")  # Remove "very_" prefix if present
        gender_suffix = "man" if speaker_gender == "male" else "woman"
        dir_name = f"{pitch_value}_pitch_{gender_suffix}"

        logging.info(f"  -> Pitch mode: {value} + {speaker_gender} → {dir_name}")

    else:
        # Map config values to directory names
        # This allows mapping "child" -> "children", "elderly" -> "old", etc.
        value_to_dir_map = {
            "child": "children",
            "elderly": "old",
            # Add more mappings as needed
        }

        # Get directory name (use mapping if exists, otherwise use value directly)
        dir_name = value_to_dir_map.get(value, value)

        if dir_name != value:
            logging.info(f"  -> Mapped '{value}' to directory '{dir_name}'")

    # Look for directory matching the tag value
    category_dir = Path(emotion_audio_pool_dir) / dir_name

    if not category_dir.exists():
        logging.warning(f"  -> ❌ Emotion reference directory not found: {category_dir} (will use default)")
        return None

    # Get all audio files in the category (.mp3 or .wav)
    audio_files = list(category_dir.glob("*.mp3")) + list(category_dir.glob("*.wav"))

    if not audio_files:
        logging.warning(f"  -> ❌ No audio files found in {category_dir} (will use default)")
        return None

    # Randomly select one audio file
    selected_audio = random.choice(audio_files)

    logging.info(f"  -> ✓ Selected emotion reference: {selected_audio.name} from {category_dir}")
    logging.info(f"     Full path: {selected_audio}")
    return str(selected_audio)

def IndexTTS_gen(script, output, tts_model, spk_audio_dir="examples", emotion_audio_pool_dir=None):
    """
    Generate TTS audio using IndexTTS2 model with emotion reference audio.
    Uses IndexTTS-2's disentangled architecture:
    - Timbre Prompt (spk_audio_prompt): Controls speaker voice identity
    - Style Prompt (emo_audio_prompt): Controls tone/style from reference audio

    Handles paralinguistic tags by mapping them to emotion reference audio files.

    Args:
        script: List of (role, text) tuples where text may contain (tags)
        output: Path to save the output wav file
        tts_model: Pre-initialized IndexTTS2 model instance (reused across all files)
        spk_audio_dir: Directory containing speaker reference audio files
        emotion_audio_pool_dir: Directory containing emotion reference audio pool
    """
    sample_rate = 24000  # IndexTTS2 default sample rate
    tts = tts_model  # Use the pre-initialized model

    # ═══════════════════════════════════════════════════════════════════════════
    # GENDER MODE PRE-DETECTION: Find first gender tag to set initial speaker
    # ═══════════════════════════════════════════════════════════════════════════
    first_gender_tag = pre_detect_first_gender_tag(script)

    # Find available speaker audio files
    all_dir = Path(spk_audio_dir) / "all"
    spk_audio_files = list(all_dir.glob("*.wav"))
    if len(spk_audio_files) < 2:
        raise ValueError(f"Need at least 2 speaker audio files in {all_dir}. Found {len(spk_audio_files)}")

    # Select User speaker (Speaker A) - always random
    spk_A_audio = str(random.choice(spk_audio_files))
    remaining = [f for f in spk_audio_files if str(f) != spk_A_audio]

    # Select Agent speaker (Speaker B) - depends on gender mode
    if first_gender_tag:
        # GENDER MODE: Set initial agent speaker to OPPOSITE of first gender tag
        initial_agent_gender = get_opposite_gender(first_gender_tag)
        if initial_agent_gender:
            logging.info(f"  ═══ GENDER MODE INITIALIZATION ═══")
            logging.info(f"  -> First gender switch will be to: {first_gender_tag}")
            logging.info(f"  -> Setting initial agent gender to: {initial_agent_gender} (opposite)")

            # Sample from the opposite gender directory
            gender_dir = Path(spk_audio_dir) / initial_agent_gender
            gender_audio_files = list(gender_dir.glob("*.wav"))
            if gender_audio_files:
                spk_B_audio = str(random.choice(gender_audio_files))
                spk_B_gender = initial_agent_gender
                logging.info(f"  -> ✓ Initial agent speaker set: {Path(spk_B_audio).name} ({initial_agent_gender})")
            else:
                logging.warning(f"  -> ❌ No audio files in {gender_dir}, using random")
                spk_B_audio = str(random.choice(remaining))
                spk_B_gender = detect_speaker_gender(spk_B_audio, spk_audio_dir)
        else:
            # Fallback to random
            spk_B_audio = str(random.choice(remaining))
            spk_B_gender = detect_speaker_gender(spk_B_audio, spk_audio_dir)
    else:
        # NO GENDER MODE: Random selection
        spk_B_audio = str(random.choice(remaining))
        spk_B_gender = detect_speaker_gender(spk_B_audio, spk_audio_dir)

    logging.info(f"Speaker A (User): {spk_A_audio}")
    logging.info(f"Speaker B (Agent): {spk_B_audio}")

    # Detect gender for User speaker (for pitch mode)
    spk_A_gender = detect_speaker_gender(spk_A_audio, spk_audio_dir)

    # Process each dialogue turn
    audio_segments = []

    # Track current speaker audio for each role (can change in gender mode)
    current_spk_A_audio = spk_A_audio
    current_spk_B_audio = spk_B_audio
    current_spk_A_gender = spk_A_gender
    current_spk_B_gender = spk_B_gender

    # Track last para tag and emotion reference audio for each role (to avoid redundant resampling)
    # This applies to ALL para tags (emotion, speed, volume, age, pitch) that use emotion reference
    last_spk_A_para_tag = None  # Format: (dimension, value) e.g., ("speed", "fast"), ("emotion", "happy")
    last_spk_A_emo_audio = None  # Last emotion reference audio path
    last_spk_B_para_tag = None
    last_spk_B_emo_audio = None

    for idx, (role, text) in enumerate(script, 1):
        # Assign speaker based on role (User = spk_A, Agent = spk_B)
        is_user = role in ("User", "[overlap] User")
        spk_audio = current_spk_A_audio if is_user else current_spk_B_audio
        spk_gender = current_spk_A_gender if is_user else current_spk_B_gender

        # Get last para tag and reference for this role
        last_para_tag = last_spk_A_para_tag if is_user else last_spk_B_para_tag
        last_emo_audio = last_spk_A_emo_audio if is_user else last_spk_B_emo_audio

        # Extract paralinguistic tags from text using existing function
        clean_text, para_tags = extract_paralinguistic_tags(text)

        # Log extracted tags
        if para_tags:
            logging.info(f"Turn {idx} ({role}): Extracted tags {para_tags}")

        # ═══════════════════════════════════════════════════════════════════════
        # PARALINGUISTIC TAG HANDLING - THREE MODES
        # ═══════════════════════════════════════════════════════════════════════
        # 1. GENDER MODE: Switch speaker reference audio (no emotion ref)
        # 2. PITCH MODE: Use gender-aware emotion reference
        # 3. OTHER MODES: Use emotion reference audio
        # ═══════════════════════════════════════════════════════════════════════

        emo_audio_prompt = None

        if para_tags:
            # Get the dimension and value (should be only ONE per utterance)
            dimension = next(iter(para_tags.keys()))
            value = para_tags[dimension]

            # ─────────────────────────────────────────────────────────────────
            # GENDER MODE: Switch speaker reference (no emotion reference)
            # ─────────────────────────────────────────────────────────────────
            if dimension == "gender":
                logging.info(f"  ═══ GENDER MODE ═══")
                logging.info(f"  -> Gender tag detected: {value}")

                # Map gender values to directory names
                gender_dir_map = {
                    "man": "male",
                    "male": "male",
                    "woman": "female",
                    "female": "female"
                }
                target_gender_dir = gender_dir_map.get(value.lower())

                # Check if we need to switch (only if different from current gender)
                if target_gender_dir and target_gender_dir != spk_gender:
                    logging.info(f"  -> Current gender: {spk_gender}, Target gender: {target_gender_dir}")
                    logging.info(f"  -> Gender is DIFFERENT - switching to {value} voice")

                    # Sample a new speaker from the target gender directory
                    gender_dir = Path(spk_audio_dir) / target_gender_dir
                    gender_audio_files = list(gender_dir.glob("*.wav"))

                    if gender_audio_files:
                        # Randomly select a speaker from the target gender
                        new_spk_audio = str(random.choice(gender_audio_files))
                        logging.info(f"  -> ✓ Selected new speaker: {Path(new_spk_audio).name}")

                        # Update current speaker audio and gender
                        spk_audio = new_spk_audio
                        spk_gender = target_gender_dir  # "male" or "female"

                        # Update tracking variables for this role
                        if is_user:
                            current_spk_A_audio = new_spk_audio
                            current_spk_A_gender = spk_gender
                        else:
                            current_spk_B_audio = new_spk_audio
                            current_spk_B_gender = spk_gender

                        logging.info(f"  -> ✓ Speaker reference updated for {role}")
                    else:
                        logging.warning(f"  -> ❌ No audio files found in {gender_dir}")
                elif target_gender_dir == spk_gender:
                    # Gender is the same - no need to switch
                    logging.info(f"  -> Current gender is already {spk_gender} - NO SWITCH NEEDED")
                else:
                    logging.warning(f"  -> ❌ Unknown gender value: {value}")

                # Do NOT use emotion reference audio in gender mode
                emo_audio_prompt = None

                # Clear para tag tracking when gender changes (since gender mode doesn't use emotion ref)
                if is_user:
                    last_spk_A_para_tag = None
                    last_spk_A_emo_audio = None
                else:
                    last_spk_B_para_tag = None
                    last_spk_B_emo_audio = None

            # ─────────────────────────────────────────────────────────────────
            # PITCH MODE: Use gender-aware emotion reference
            # ─────────────────────────────────────────────────────────────────
            elif dimension == "pitch":
                logging.info(f"  ═══ PITCH MODE ═══")

                # Check if para tag is the same as last time (to avoid redundant resampling)
                current_para_tag = (dimension, value)
                if last_para_tag == current_para_tag and last_emo_audio:
                    # Same para tag - REUSE the same emotion reference audio
                    emo_audio_prompt = last_emo_audio
                    logging.info(f"  -> Para tag unchanged ({dimension}={value}) - REUSING emotion reference")
                    logging.info(f"  -> ✓ Reusing: {Path(emo_audio_prompt).name}")
                else:
                    # Different para tag - resample new emotion reference audio
                    if emotion_audio_pool_dir:
                        # Pass speaker gender to get_emotion_reference_audio
                        emo_audio_prompt = get_emotion_reference_audio(
                            para_tags, emotion_audio_pool_dir, speaker_gender=spk_gender
                        )
                        if emo_audio_prompt:
                            logging.info(f"  -> ✓ Will apply {dimension}={value} using gender-aware emotion reference")
                            # Update tracking for this role
                            if is_user:
                                last_spk_A_para_tag = current_para_tag
                                last_spk_A_emo_audio = emo_audio_prompt
                            else:
                                last_spk_B_para_tag = current_para_tag
                                last_spk_B_emo_audio = emo_audio_prompt
                        else:
                            logging.warning(f"  -> ❌ Could not find audio for {dimension}:{value}, using default voice")
                    else:
                        logging.warning(f"  -> ⚠️  emotion_audio_pool_dir not set, skipping {dimension}:{value}")

            # ─────────────────────────────────────────────────────────────────
            # OTHER MODES: Use emotion reference audio (emotion, speed, volume, age)
            # ─────────────────────────────────────────────────────────────────
            else:
                logging.info(f"  ═══ EMOTION REFERENCE AUDIO MODE ═══")

                # Check if para tag is the same as last time (to avoid redundant resampling)
                current_para_tag = (dimension, value)
                if last_para_tag == current_para_tag and last_emo_audio:
                    # Same para tag - REUSE the same emotion reference audio
                    emo_audio_prompt = last_emo_audio
                    logging.info(f"  -> Para tag unchanged ({dimension}={value}) - REUSING emotion reference")
                    logging.info(f"  -> ✓ Reusing: {Path(emo_audio_prompt).name}")
                else:
                    # Different para tag - resample new emotion reference audio
                    if emotion_audio_pool_dir:
                        emo_audio_prompt = get_emotion_reference_audio(para_tags, emotion_audio_pool_dir)
                        if emo_audio_prompt:
                            logging.info(f"  -> ✓ Will apply {dimension}={value} using emotion reference audio")
                            # Update tracking for this role
                            if is_user:
                                last_spk_A_para_tag = current_para_tag
                                last_spk_A_emo_audio = emo_audio_prompt
                            else:
                                last_spk_B_para_tag = current_para_tag
                                last_spk_B_emo_audio = emo_audio_prompt
                        else:
                            logging.warning(f"  -> ❌ Could not find audio for {dimension}:{value}, using default voice")
                    else:
                        logging.info(f"  -> ⚠️  emotion_audio_pool_dir not set, skipping {dimension}:{value}")

        # Generate temporary output path for this turn
        temp_output = f"/tmp/indextts_turn_{idx}.wav"

        # Generate speech with IndexTTS2
        try:
            # ═══════════════════════════════════════════════════════════════════════
            # TTS INFERENCE WITH EMOTION REFERENCE AUDIO
            # ═══════════════════════════════════════════════════════════════════════
            # Build inference kwargs
            infer_kwargs = {
                "spk_audio_prompt": spk_audio,  # Timbre prompt (speaker identity)
                "text": clean_text,
                "output_path": temp_output,
                "use_random": False,
                "verbose": False
            }

            # Add emotion reference audio if available
            if emo_audio_prompt is not None:
                # Try to use emotion reference audio parameter
                # This assumes IndexTTS2 has a parameter for emotion reference audio
                # You may need to adjust the parameter name based on actual IndexTTS2 API
                try:
                    infer_kwargs["emo_audio_prompt"] = emo_audio_prompt
                    logging.info(f"  -> 🎵 Passing emotion reference to TTS model (emo_audio_prompt)")
                except TypeError:
                    # If parameter not supported, try alternative names
                    logging.warning("  -> ⚠️  emo_audio_prompt parameter not supported, trying style_audio_prompt")
                    try:
                        infer_kwargs["style_audio_prompt"] = emo_audio_prompt
                        logging.info(f"  -> 🎵 Passing emotion reference to TTS model (style_audio_prompt)")
                    except TypeError:
                        logging.error("  -> ❌ Emotion reference audio not supported by this IndexTTS2 version")
                        # Fall back to using only speaker prompt
                        pass

            tts.infer(**infer_kwargs)

            # Load generated audio
            wav, sr = torchaudio.load(temp_output)

            # Resample if necessary
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)

            # Ensure mono (take first channel if stereo)
            if wav.shape[0] > 1:
                wav = wav[0:1, :]

            # ═══════════════════════════════════════════════════════════════════════
            # POST-PROCESSING for volume (and other tags if needed)
            # ═══════════════════════════════════════════════════════════════════════
            # Volume requires BOTH emotion reference AND post-processing:
            # - Emotion reference provides style/tone example to TTS model
            # - Post-processing adjusts actual dB level via amplitude scaling
            # ═══════════════════════════════════════════════════════════════════════
            if para_tags:
                try:
                    wav = apply_audio_post_processing(wav, sample_rate, para_tags)
                    logging.info(f"  -> ✓ Post-processing applied successfully")
                except Exception as post_error:
                    logging.error(f"  -> ❌ Post-processing failed: {post_error}")
                    logging.error(f"     Error type: {type(post_error).__name__}")
                    logging.error(f"     Keeping original audio without post-processing")
                    # Keep original wav without post-processing instead of failing

            # Store with role information for stereo placement
            audio_segments.append((role, wav, spk_A_audio))

            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)

        except Exception as e:
            logging.error(f"Error generating audio for turn {idx}: {e}")
            # Create silence as fallback
            silence = torch.zeros(1, sample_rate)
            audio_segments.append((role, silence, spk_A_audio))

    # Combine audio segments into stereo (User=left, Agent=right)
    l_ch = None
    r_ch = None

    for idx, (role, wav, spk_ref) in enumerate(audio_segments):
        is_user = role in ("User", "[overlap] User")
        is_overlap = role.strip().lower().startswith("[overlap]")

        if idx == 0:
            # First utterance
            if is_user:
                l_ch = wav
                r_ch = torch.zeros_like(wav)
            else:
                r_ch = wav
                l_ch = torch.zeros_like(wav)
        else:
            if is_overlap:
                # Handle overlap - reduce previous audio and overlap
                overlap_frame = int(random.uniform(0.6, 1.0) * sample_rate)
                padded = torch.zeros(1, wav.shape[-1] - overlap_frame)
                pause = torch.zeros(1, sample_rate // 4)

                if is_user:
                    l_ch = l_ch[:, :-overlap_frame]
                    l_ch = torch.cat([l_ch, pause, wav], -1)
                    r_ch = torch.cat([r_ch, pause, padded], -1)
                else:
                    r_ch = r_ch[:, :-overlap_frame]
                    r_ch = torch.cat([r_ch, pause, wav], -1)
                    l_ch = torch.cat([l_ch, pause, padded], -1)
            else:
                # Normal concatenation with pause
                padded = torch.zeros_like(wav)
                pause = torch.zeros(1, sample_rate // 4)

                if is_user:
                    l_ch = torch.cat([l_ch, pause, wav], -1)
                    r_ch = torch.cat([r_ch, pause, padded], -1)
                else:
                    r_ch = torch.cat([r_ch, pause, wav], -1)
                    l_ch = torch.cat([l_ch, pause, padded], -1)

    # Combine stereo channels
    full_dialog = torch.cat([l_ch, r_ch], dim=0)

    # Save audio file
    torchaudio.save(str(output), full_dialog, sample_rate)
    logging.info(f"Saved TTS audio to: {output}")

def tts_batch(cfg):
    """
    Batch process dialogue text files to generate TTS audio using IndexTTS2.
    Processes files from dialogue output directory and extracts paralinguistic tags.
    Uses dual-node approach:
    - Emotion tags → Emotion vectors
    - Gender/Age tags → Emotion reference audio
    """
    src_dir = Path(cfg.dialogue["out_dir"])
    wav_dir = Path(cfg.tts["wav_dir"])
    wav_dir.mkdir(parents=True, exist_ok=True)

    # IndexTTS2 model directory and speaker audio directory from config
    model_dir = cfg.tts.get("model_dir", "checkpoints")
    spk_audio_dir = cfg.tts.get("spk_audio_dir", "examples")
    emotion_audio_pool_dir = cfg.tts.get("emotion_audio_pool_dir", None)  # NEW: Emotion reference audio pool

    # Create error log file
    error_log_path = wav_dir / "tts_errors.txt"

    # Get all dialogue text files
    dialogue_files = list(src_dir.glob("*.txt"))

    if not dialogue_files:
        logging.warning(f"No dialogue files found in {src_dir}")
        return

    logging.info(f"Processing {len(dialogue_files)} dialogue files for TTS...")

    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZE IndexTTS2 MODEL ONCE (REUSE FOR ALL FILES)
    # ═══════════════════════════════════════════════════════════════════════════
    logging.info("Initializing IndexTTS2 model with optimizations...")
    cfg_path = os.path.join(model_dir, "config.yaml")

    # Check CUDA availability before initialization
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")
        logging.info(f"Current GPU: {torch.cuda.current_device()}")
        logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")

    if not torch.cuda.is_available():
        logging.error("=" * 80)
        logging.error("CUDA is NOT available! TTS will run on CPU (VERY SLOW)")
        logging.error("Please install PyTorch with CUDA support:")
        logging.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        logging.error("=" * 80)
        raise RuntimeError("CUDA is required but not available. Please check your PyTorch installation.")

    tts_model = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",  # FORCE GPU usage (will error if CUDA not available)
        use_fp16=True,  # ENABLED: 2-3x faster inference with less VRAM
        use_cuda_kernel=True,  # ENABLED: Faster BigVGAN vocoder with custom CUDA kernel
        use_accel=False,  # DISABLED: Requires flash_attn (not installed)
        use_torch_compile=False,  # DISABLED: Can cause compatibility issues
        use_deepspeed=False  # DeepSpeed disabled (requires extra setup)
    )
    logging.info("IndexTTS2 model initialized successfully with all optimizations!")

    for txt_file in tqdm(dialogue_files, desc="TTS Generation"):
        # Output wav file path
        wav_file_path = wav_dir / f"{txt_file.stem}_IndexTTS.wav"

        # Skip if already exists
        if wav_file_path.exists():
            logging.info(f"Skipping {txt_file.name} - wav file already exists")
            continue

        try:
            # Read dialogue file
            text = txt_file.read_text("utf-8")
            script = []

            # Parse dialogue into (role, text) tuples
            for line in text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    role, content = line.split(":", 1)
                    script.append((role.strip(), content.strip()))

            if not script:
                logging.warning(f"No dialogue turns found in {txt_file.name}")
                continue

            # Generate TTS audio
            logging.info(f"Generating TTS for {txt_file.name}...")
            IndexTTS_gen(
                script,
                wav_file_path,
                tts_model=tts_model,  # Pass pre-initialized model
                spk_audio_dir=spk_audio_dir,
                emotion_audio_pool_dir=emotion_audio_pool_dir  # Pass emotion audio pool
            )

        except Exception as e:
            # Log error and continue with next file
            error_msg = f"Error processing {txt_file.name}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            with open(error_log_path, "a", encoding="utf-8") as error_file:
                error_file.write(f"{txt_file.name}: {str(e)}\n")
            continue

    logging.info(f"TTS batch processing complete. Audio files saved to: {wav_dir}")

# ─────────────────────────────  ORCHESTRATOR  ──────────────────────────────

@dataclass
class PipelineConfig:
    # device
    device: str = "cuda"

    # stages to run
    stages: List[str] = field(default_factory=lambda: [
        "scenario", "dialogue", "analysis", "tts"])

    # scenario
    scenario: OmegaConf = OmegaConf.create({
        "model": "openai/gpt-4o-mini",
        "n": 10,
        "out_file": "data_para/scenarios.json",
        "prompt": "Generate {n} diverse customer service scenarios in JSON format."
    })

    # dialogue (PARALINGUISTIC VERSION)
    dialogue: OmegaConf = OmegaConf.create({
        "user_model": "meta-llama/llama-3.3-70b-instruct",
        "agent_model": "openai/gpt-4o-mini",
        "user_prompt": "You are a customer/user interacting with a support agent. Be natural and conversational.",
        "agent_prompt": "You are a helpful customer support agent. Provide clear and professional responses.",
        "out_dir": "data_para/dialog_txt_para",
        "min_turns": 4,
        "max_turns": 10,
        "per_scenario": 3,
        "use_paralinguistic": True,
        "paralinguistic_probability": 1.0,  # 100% of agent turns will have paralinguistic tags
    })

    # paralinguistic_pools (loaded from config file)
    paralinguistic_pools: OmegaConf = OmegaConf.create({})

    # tts (IndexTTS2)
    tts: OmegaConf = OmegaConf.create({
        "wav_dir": "data_para/tts_audio",
        "model_dir": "checkpoints",
        "spk_audio_dir": "examples",
    })

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        print("DEBUG stages =", self.cfg.stages)

    def run(self):
        st = set(self.cfg.stages)

        if "scenario" in st:
            print("Scenario Creating...")
            generate_scenarios(self.cfg)
            original_path = Path(self.cfg.scenario["out_file"])
            convert_nested_json_to_jsonl(original_path, original_path.with_suffix(".jsonl"))

        if "dialogue" in st:
            print("Dialogue Generating (PARALINGUISTIC VERSION)...")
            generate_paralinguistic_dialogues(self.cfg)

        if "analysis" in st:
            print("Analyzing Paralinguistic Distribution...")
            analyze_paralinguistic_distribution(self.cfg)

        if "tts" in st:
            print("Generating TTS Audio (IndexTTS2)...")
            tts_batch(self.cfg)

# ─────────────────────────────  CLI ENTRY  ────────────────────────────────

def main():  # pragma: no cover
    @hydra.main(config_path="conf", config_name="base_para", version_base=None)
    def _run(cfg):
        if not isinstance(cfg, PipelineConfig):
            cfg = OmegaConf.merge(OmegaConf.structured(PipelineConfig), cfg)

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        Pipeline(cfg).run()

    print("Finish Config Matching")
    _run()

if __name__ == "__main__":
    main()

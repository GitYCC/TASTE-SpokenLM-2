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

# Load environment variables from .env file
load_dotenv()

APIKEY = os.getenv("OPENROUTER_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════
# TTS INITIALIZATION - XTTS2
# ═══════════════════════════════════════════════════════════════════════════

# Import TTS model initialization
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])
from TTS.api import TTS

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

def select_control_target(pools: Dict[str, List[str]], fixed_dimension: str) -> Tuple[str, str]:
    """
    Select a paralinguistic target value for the specified dimension.

    Args:
        pools: Paralinguistic dimension pools from config
        fixed_dimension: The specific dimension to use (from config)

    Returns: (dimension, target_value)
             e.g., ("speed", "fast") or ("emotion", "happy")
    """
    # Use the fixed dimension from config
    dimension = fixed_dimension
    target_value = random.choice(pools[dimension])
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
    control_request_frequency = cfg.dialogue.get("control_request_frequency", 0.3)  # 30% chance per user turn
    control_dimension = cfg.dialogue.get("control_dimension", "speed")  # Fixed dimension from config

    # Load paralinguistic pools from config (single source of truth)
    paralinguistic_pools = cfg.get("paralinguistic_pools", {})

    logging.info(f"Generating PARALINGUISTIC dialogues:")
    logging.info(f"  User Model: {user_model}")
    logging.info(f"  Agent Model: {agent_model}")
    logging.info(f"  Turn Range: {min_turns}-{max_turns}")
    logging.info(f"  Dialogues per Scenario: {dialogues_per_scenario}")
    logging.info(f"  Control Request Frequency: {control_request_frequency}")
    logging.info(f"  Control Dimension (Fixed): {control_dimension}")

    for scenario in tqdm(scenarios, desc="dialogues"):
        scenario_desc = json.dumps(scenario, ensure_ascii=False)

        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_turns = []
            conversation_history = []

            context_info = f"Scenario: {scenario_desc}\n\n"
            num_turns = random.randint(min_turns, max_turns)
            logging.info(f"  Generating {scenario['id']}_{dialogue_idx + 1} with {num_turns} turns")

            for turn in range(num_turns):
                if turn % 2 == 0:
                    # ============ USER's TURN ============
                    # RANDOM SWITCH: Decide if this user turn should include a control request
                    # Need to ensure agent has next turn to respond (turn < num_turns - 1)
                    can_make_control_request = turn < num_turns - 1
                    is_control_request = can_make_control_request and random.random() < control_request_frequency
                    control_request_info = None

                    if is_control_request:
                        # Select a control target value for the fixed dimension
                        ctrl_dim, ctrl_value = select_control_target(paralinguistic_pools, control_dimension)
                        control_request_info = (ctrl_dim, ctrl_value)
                        logging.info(f"    [CONTROL] Turn {turn}: Requesting {ctrl_dim}={ctrl_value}")

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

                        # Reset control request after processing
                        control_request_info = None

                    else:
                        # USUAL MODE - Normal agent response WITHOUT tags
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

                        # For conversation history, use clean version (without tags)
                        clean_utterance, _ = extract_paralinguistic_tags(agent_utterance)
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

    # Apply SPEED modification (time-stretch)
    if "speed" in para_tags:
        speed_map = {
            "very_slow": 0.5,
            "slow": 0.75,
            "normal": 1.0,
            "fast": 1.25,
            "very_fast": 1.5
        }
        speed_factor = speed_map.get(para_tags["speed"], 1.0)

        if speed_factor != 1.0:
            logging.info(f"  -> Applying speed: {para_tags['speed']} (factor: {speed_factor})")
            # Process each channel
            processed_channels = []
            for ch in audio_np:
                # Time-stretch without changing pitch
                stretched = librosa.effects.time_stretch(ch, rate=speed_factor)
                processed_channels.append(stretched)
            audio_np = np.array(processed_channels)

    # Apply PITCH modification (pitch-shift)
    if "pitch" in para_tags:
        pitch_map = {
            "very_low": -4,     # semitones
            "low": -2,
            "normal": 0,
            "high": 2,
            "very_high": 4
        }
        pitch_shift = pitch_map.get(para_tags["pitch"], 0)

        if pitch_shift != 0:
            logging.info(f"  -> Applying pitch: {para_tags['pitch']} (shift: {pitch_shift} semitones)")
            # Process each channel
            processed_channels = []
            for ch in audio_np:
                # Pitch-shift
                shifted = librosa.effects.pitch_shift(ch, sr=sample_rate, n_steps=pitch_shift)
                processed_channels.append(shifted)
            audio_np = np.array(processed_channels)

    # Apply VOLUME modification (amplitude scaling)
    if "volume" in para_tags:
        volume_map = {
            "very_quiet": 0.3,
            "quiet": 0.6,
            "normal": 1.0,
            "loud": 1.5,
            "very_loud": 2.0
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

def XTTS_gen(script, output, use_gpu=False):
    """
    Generate TTS audio using XTTS2 model.
    Handles paralinguistic tags by extracting them from text.

    Args:
        script: List of (role, text) tuples where text may contain (tags)
        output: Path to save the output wav file
        use_gpu: Whether to use GPU for TTS inference
    """
    sample_rate = 22050
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

    # Speaker pools (same as syn_ver_1.py)
    male_speakers = ['Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack',
                     'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black',
                     'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Torcull Diarmuid',
                     'Viktor Menelaos', 'Zacharie Aimilios', 'Ige Behringer', 'Filip Traverse',
                     'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl',
                     'Eugenio Mataracı', 'Xavier Hayasaka', 'Luis Moray']

    female_speakers = ['Alison Dietlinde', 'Alexandra Hisakawa', 'Alma María', 'Ana Florence',
                       'Asya Anara', 'Andrew Chipper', 'Annmarie Nele', 'Badr Odhiambo',
                       'Barbora MacLean', 'Brenda Stern', 'Camilla Holmström', 'Chandra MacFarland',
                       'Claribel Dervla', 'Daisy Studious', 'Gitta Nikolina', 'Gracie Wise',
                       'Henriette Usha', 'Lidiya Szekeres', 'Lilya Stainthorpe', 'Maja Ruoho',
                       'Nova Hogarth', 'Narelle Moon', 'Rosemary Okafor', 'Sofia Hellen',
                       'Szofi Granger', 'Suad Qasim', 'Tammie Ema', 'Tammy Grit', 'Tanja Adelina',
                       'Uta Obando', 'Vjollca Johnnie']

    all_speakers = male_speakers + female_speakers
    spk_A = random.choice(all_speakers)
    while True:
        spk_B = random.choice(all_speakers)
        if spk_B != spk_A:
            break

    for idx, (role, text) in enumerate(script, 1):
        # Assign speaker based on role (User = spk_A, Agent = spk_B)
        spk_id = spk_A if role in ("User", "[overlap] User") else spk_B

        # Extract paralinguistic tags from text using existing function
        clean_text, para_tags = extract_paralinguistic_tags(text)

        # Log extracted tags
        if para_tags:
            logging.info(f"Turn {idx} ({role}): Extracted tags {para_tags}")

        # Generate speech with XTTS2 (no paralinguistic control at generation time)
        wav = tts.tts(
            text=clean_text,
            speaker=spk_id,
            language="en",
        )

        # Convert to tensor if needed
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Apply post-processing based on paralinguistic tags
        if para_tags:
            wav = apply_audio_post_processing(wav, sample_rate, para_tags)

        # Create stereo channels
        if idx == 1:
            # First utterance
            l_ch = wav
            r_ch = torch.zeros_like(wav)
        else:
            if role.strip().lower().startswith("[overlap]"):
                # Handle overlap
                overlap_frame = int(random.uniform(0.6, 1.0) * sample_rate)
                padded = torch.zeros(1, wav.shape[-1] - overlap_frame)
                if spk_id == spk_A:
                    l_ch = l_ch[:, :-overlap_frame]
                    pause = torch.zeros(1, sample_rate // 4)
                    l_ch = torch.cat([l_ch, pause, wav], -1)
                    r_ch = torch.cat([r_ch, pause, padded], -1)
                else:
                    r_ch = r_ch[:, :-overlap_frame]
                    pause = torch.zeros(1, sample_rate // 4)
                    r_ch = torch.cat([r_ch, pause, wav], -1)
                    l_ch = torch.cat([l_ch, pause, padded], -1)
            else:
                # Normal concatenate with pause
                padded = torch.zeros_like(wav)
                pause = torch.zeros(1, sample_rate // 4)
                if spk_id == spk_A:
                    l_ch = torch.cat([l_ch, pause, wav], -1)
                    r_ch = torch.cat([r_ch, pause, padded], -1)
                else:
                    r_ch = torch.cat([r_ch, pause, wav], -1)
                    l_ch = torch.cat([l_ch, pause, padded], -1)

    # Combine stereo channels
    full_dialog = torch.cat([l_ch, r_ch], dim=0)

    # Save audio file
    torchaudio.save(output, full_dialog, sample_rate)
    logging.info(f"Saved TTS audio to: {output}")

def tts_batch(cfg):
    """
    Batch process dialogue text files to generate TTS audio using XTTS2.
    Processes files from dialogue output directory and extracts paralinguistic tags.
    """
    src_dir = Path(cfg.dialogue["out_dir"])
    wav_dir = Path(cfg.tts["wav_dir"])
    wav_dir.mkdir(parents=True, exist_ok=True)

    # GPU setting from config
    use_gpu = cfg.get("device", "cuda") == "cuda"

    # Create error log file
    error_log_path = wav_dir / "tts_errors.txt"

    # Get all dialogue text files
    dialogue_files = list(src_dir.glob("*.txt"))

    if not dialogue_files:
        logging.warning(f"No dialogue files found in {src_dir}")
        return

    logging.info(f"Processing {len(dialogue_files)} dialogue files for TTS...")

    for txt_file in tqdm(dialogue_files, desc="TTS Generation"):
        # Output wav file path
        wav_file_path = wav_dir / f"{txt_file.stem}_XTTS.wav"

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
            XTTS_gen(script, wav_file_path, use_gpu=use_gpu)

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

    # tts (XTTS2)
    tts: OmegaConf = OmegaConf.create({
        "wav_dir": "data_para/tts_audio",
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
            print("Generating TTS Audio (XTTS2)...")
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

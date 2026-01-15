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

# Load environment variables from .env file
load_dotenv()

APIKEY = os.getenv("OPENROUTER_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════
# TTS INITIALIZATION - IndexTTS2
# ═══════════════════════════════════════════════════════════════════════════

# Import IndexTTS2 model
from indextts.infer_v2 import IndexTTS2

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTATION MODE - AGENT ADAPTS TO USER'S PARALINGUISTIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════
# This file implements ADAPTATION MODE where:
# - USER has FIXED paralinguistic features throughout entire dialogue
#   (Age, Emotion, Gender, Sarcasm) - determined at dialogue start
# - AGENT adapts its response based on user's FIXED para tags
# - BOTH user and agent maintain their respective para tags from START to END
#
# Example dialogue flow:
#   User is "sad" (fixed) → Agent responds "calm" (fixed adaptive)
#   Turn 1: User: (emotion:sad) I'm disappointed...
#           Agent: (emotion:calm) I understand your concern...
#   Turn 2: User: (emotion:sad) This is frustrating...
#           Agent: (emotion:calm) Let me help you with that...
#   ... (maintains throughout entire dialogue)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTATION MAPPING - How agent should respond to user's paralinguistic features
# ═══════════════════════════════════════════════════════════════════════════
#
# NOTE: For EMOTION dimension, the agent LLM now intelligently selects its emotion
# based on full dialogue context (scenario + user emotion + available emotions).
# This happens during the FIRST agent turn - no hardcoded mapping needed!
#
# For other dimensions (age, gender, sarcasm), adaptation is handled via prompt
# instructions rather than paralinguistic tags.
# ═══════════════════════════════════════════════════════════════════════════

AGE_ADAPTATION_MAP = {
    # User age → Agent response style (encoded in prompt, not in tags)
    "child": "patient and gentle with simple language",
    "elderly": "patient and respectful with clear communication",
    "young_adult": "professional and friendly",
    "middle_aged": "professional and efficient",
}

# NOTE: Gender and Sarcasm don't require paralinguistic tag adaptation
# They are handled in the agent's text generation prompt

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

# ─────────────────────────────  ADAPTATION MODE UTILS  ──────────────────────────────
#
# Agent emotion selection is now handled directly in the dialogue generation loop.
# The agent LLM selects its emotion on the FIRST turn based on:
#   - Full scenario context
#   - User's fixed emotion
#   - All available emotions from config
# This selected emotion is then maintained throughout the entire dialogue.
#
# No separate utility function needed - emotion selection is integrated into
# the agent's response generation for maximum efficiency (no extra API calls).
# ═══════════════════════════════════════════════════════════════════════════

def get_adaptation_instruction(user_para_tags: Optional[Dict[str, str]]) -> str:
    """
    Generate instruction for agent to adapt to user's paralinguistic features.

    Args:
        user_para_tags: User's paralinguistic tags (e.g., {"emotion": "happy"})

    Returns:
        Instruction text for agent's LLM prompt
    """
    if not user_para_tags:
        return ""

    dimension = next(iter(user_para_tags.keys()))
    value = user_para_tags[dimension]

    instruction_parts = []

    # Emotion-based adaptation
    if dimension == "emotion":
        emotion_guidance = {
            "happy": "The user sounds happy and cheerful. Respond with enthusiasm and positive energy.",
            "sad": "The user sounds sad. Respond with empathy, gentleness, and emotional support.",
            "angry": "The user sounds angry or frustrated. Stay calm, apologetic, and understanding. Try to de-escalate.",
            "afraid": "The user sounds afraid or anxious. Respond with calm reassurance and gentle support.",
            "disgusted": "The user sounds disgusted or upset. Stay calm and understanding without judgment.",
            "surprised": "The user sounds surprised. Respond helpfully and positively.",
            "calm": "The user sounds calm and composed. Mirror their calmness with professionalism.",
            "melancholic": "The user sounds melancholic or down. Respond with empathy and gentle support.",
        }
        instruction_parts.append(emotion_guidance.get(value, "Respond naturally to the user's tone."))

    # Age-based adaptation
    elif dimension == "age":
        age_guidance = {
            "child": "The user is a child. Use simple, clear language. Be patient and gentle.",
            "elderly": "The user is elderly. Be respectful, patient, and speak clearly.",
            "young_adult": "The user is a young adult. Be professional and friendly.",
            "middle_aged": "The user is middle-aged. Be professional and efficient.",
        }
        instruction_parts.append(age_guidance.get(value, "Respond appropriately to the user."))

    # Gender-based adaptation (minimal - just awareness)
    elif dimension == "gender":
        instruction_parts.append(f"The user is {value}. Be respectful and professional.")

    # Sarcasm-based adaptation
    elif dimension == "sarcasm":
        if value == "sarcastic":
            instruction_parts.append("The user is being sarcastic. Stay professional, don't take offense, and respond helpfully.")
        else:
            instruction_parts.append("The user is speaking straightforwardly. Respond naturally.")

    return "\n".join(instruction_parts)

# ─────────────────────────────  DYNAMIC EMOTION DIALOGUE GEN  ──────────────────────────────

def generate_adaptation_dialogues(cfg):
    """
    DYNAMIC EMOTION MODE: Generates dialogues where emotions can SHIFT based on dialogue history.

    Key features:
    1. First turn: User gets initial emotion (random from pool)
    2. First agent turn: Agent selects emotion based on user's emotion + scenario
    3. Subsequent turns: BOTH user and agent select emotion dynamically based on:
       - Full dialogue history (with emotion tags visible)
       - Scenario context
       - Natural emotional progression
    4. Emotions can shift naturally (e.g., angry → calm → happy)
    5. Saves dialogue as simple .txt file (TTS extracts tags later)
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
    max_turns = cfg.dialogue.get("max_turns", 10)
    min_turns = cfg.dialogue.get("min_turns", 4)
    dialogues_per_scenario = cfg.dialogue.get("per_scenario", 3)

    # Dynamic emotion mode configuration
    user_para_dimension = cfg.dialogue.get("user_para_dimension", "emotion")

    # Load paralinguistic pools from config
    paralinguistic_pools = cfg.get("paralinguistic_pools", {})
    available_emotions = paralinguistic_pools.get("emotion", ["calm", "happy", "sad", "angry"])
    emotion_list = ", ".join(available_emotions)

    logging.info(f"Generating DYNAMIC EMOTION MODE dialogues:")
    logging.info(f"  User Model: {user_model}")
    logging.info(f"  Agent Model: {agent_model}")
    logging.info(f"  Turn Range: {min_turns}-{max_turns}")
    logging.info(f"  Dialogues per Scenario: {dialogues_per_scenario}")
    logging.info(f"  Emotion Mode: DYNAMIC (can shift based on dialogue history)")
    logging.info(f"  Available Emotions: {emotion_list}")

    for scenario in tqdm(scenarios, desc="dialogues"):
        scenario_desc = json.dumps(scenario, ensure_ascii=False)

        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_turns = []
            # conversation_history now includes emotion tags to track emotional progression
            conversation_history_with_tags = []

            context_info = f"Scenario: {scenario_desc}\n\n"
            num_turns = random.randint(min_turns, max_turns)
            # Ensure num_turns is even (dialogues must end with agent response)
            if num_turns % 2 != 0:
                num_turns += 1
            logging.info(f"  Generating {scenario['id']}_{dialogue_idx + 1} with {num_turns} turns (DYNAMIC EMOTIONS)")

            # Track current emotions for logging
            current_user_emotion = None
            current_agent_emotion = None

            for turn in range(num_turns):
                if turn % 2 == 0:
                    # ============ USER's TURN (DYNAMIC emotion selection) ============

                    if turn == 0:
                        # First turn - randomly select initial emotion
                        initial_emotion = random.choice(available_emotions)

                        para_instruction = (
                            f"\n\nIMPORTANT - EMOTION TAG:\n"
                            f"You are starting this conversation with emotion='{initial_emotion}'.\n"
                            f"Your response MUST start with the emotion tag in parentheses:\n"
                            f"(emotion:{initial_emotion})\n\n"
                            f"Format: (emotion:{initial_emotion}) [your greeting/first message]\n"
                            f"Your message should naturally reflect this emotion.\n"
                        )

                        user_instruction = (
                            f"{context_info}"
                            f"This is the start of the conversation. "
                            f"Generate the user's first message based on the scenario. "
                            f"Be natural and conversational."
                            f"{para_instruction}"
                            f"Only output the user's utterance with the emotion tag, nothing else."
                        )
                        logging.info(f"    [USER] Turn {turn}: Initial emotion = {initial_emotion}")

                    else:
                        # Subsequent turns - LLM selects emotion based on dialogue history
                        history_text = "\n".join(conversation_history_with_tags)

                        para_instruction = (
                            f"\n\nIMPORTANT - DYNAMIC EMOTION SELECTION:\n"
                            f"Based on the conversation history and how the agent responded, "
                            f"select the MOST APPROPRIATE emotion for this turn.\n\n"
                            f"Consider:\n"
                            f"- How has the conversation progressed?\n"
                            f"- Did the agent's response help or frustrate you?\n"
                            f"- What emotion would naturally follow from this interaction?\n\n"
                            f"Available emotions: {emotion_list}\n\n"
                            f"Your response MUST start with your chosen emotion tag:\n"
                            f"Format: (emotion:YOUR_CHOSEN_EMOTION) [your message]\n"
                            f"Example: (emotion:calm) Thank you, that helps clarify things.\n\n"
                            f"The emotion can be DIFFERENT from your previous turn if the conversation warrants it.\n"
                        )

                        user_instruction = (
                            f"{context_info}"
                            f"Conversation so far (with emotion tags showing emotional progression):\n{history_text}\n\n"
                            f"Generate the user's next response based on the agent's last message. "
                            f"Be natural and conversational."
                            f"{para_instruction}"
                            f"Only output the user's utterance with the emotion tag, nothing else."
                        )
                        logging.info(f"    [USER] Turn {turn}: Selecting emotion dynamically based on history")

                    user_messages = [
                        {"role": "system", "content": user_system_prompt},
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
                        # Extract emotion tag from LLM response
                        clean_utterance, llm_generated_tags = extract_paralinguistic_tags(user_utterance)

                        if llm_generated_tags and "emotion" in llm_generated_tags:
                            selected_emotion = llm_generated_tags["emotion"]
                            # Validate emotion is in available list
                            if selected_emotion not in available_emotions:
                                logging.warning(f"    [INVALID EMOTION] User LLM selected '{selected_emotion}', defaulting to 'calm'")
                                selected_emotion = "calm"
                        else:
                            # No emotion tag - use initial or default
                            if turn == 0:
                                selected_emotion = initial_emotion
                            else:
                                selected_emotion = current_user_emotion if current_user_emotion else "calm"
                            logging.warning(f"    [NO EMOTION TAG] User LLM didn't provide tag, using '{selected_emotion}'")

                        # Track emotion shift
                        if current_user_emotion and current_user_emotion != selected_emotion:
                            logging.info(f"    [EMOTION SHIFT] User: {current_user_emotion} → {selected_emotion}")
                        current_user_emotion = selected_emotion

                        # Build final utterance with validated emotion tag
                        user_para_tag_str = f"(emotion:{selected_emotion})"
                        user_utterance = f"{user_para_tag_str} {clean_utterance}"

                        dialogue_turns.append(f"User: {user_utterance}")
                        # Include emotion tag in history so LLM can see emotional progression
                        conversation_history_with_tags.append(f"User: {user_utterance}")

                        logging.info(f"    [USER EMOTION] Turn {turn}: {selected_emotion}")
                    else:
                        logging.error(f"Could not generate user response after {max_retries} retries, ending dialogue early")
                        break

                else:
                    # ============ AGENT's TURN (DYNAMIC emotion selection based on user + history) ============

                    history_text = "\n".join(conversation_history_with_tags)

                    # Get the user's current emotion for context
                    user_emotion_context = current_user_emotion if current_user_emotion else "unknown"

                    para_instruction = (
                        f"\n\nIMPORTANT - DYNAMIC EMOTION SELECTION:\n"
                        f"Based on the conversation history and the user's current emotional state, "
                        f"select the MOST APPROPRIATE emotion for your response.\n\n"
                        f"Consider:\n"
                        f"- The user's current emotion: '{user_emotion_context}'\n"
                        f"- How the conversation has progressed\n"
                        f"- What emotion would be most helpful/appropriate as a support agent?\n"
                        f"- Your goal: provide excellent customer support and help the user\n\n"
                        f"Available emotions: {emotion_list}\n\n"
                        f"Your response MUST start with your chosen emotion tag:\n"
                        f"Format: (emotion:YOUR_CHOSEN_EMOTION) [your response]\n"
                        f"Example: (emotion:calm) I understand your concern, let me help you with that.\n\n"
                        f"The emotion can CHANGE from your previous turn based on how the user's emotion evolves.\n"
                        f"For example:\n"
                        f"- If user is angry → you might choose 'calm' to de-escalate\n"
                        f"- If user becomes happy → you might choose 'happy' to match their positive energy\n"
                        f"- If user is sad → you might choose 'calm' with empathy\n"
                    )

                    agent_instruction = (
                        f"{context_info}"
                        f"Conversation so far (with emotion tags showing emotional progression):\n{history_text}\n\n"
                        f"Generate the agent's response to the user's last message. "
                        f"Be helpful, professional, and natural."
                        f"{para_instruction}"
                        f"Only output the agent's utterance with the emotion tag, nothing else."
                    )

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
                        # Extract emotion tag from LLM response
                        clean_utterance, llm_generated_tags = extract_paralinguistic_tags(agent_utterance)

                        if llm_generated_tags and "emotion" in llm_generated_tags:
                            selected_emotion = llm_generated_tags["emotion"]
                            # Validate emotion is in available list
                            if selected_emotion not in available_emotions:
                                logging.warning(f"    [INVALID EMOTION] Agent LLM selected '{selected_emotion}', defaulting to 'calm'")
                                selected_emotion = "calm"
                        else:
                            # No emotion tag - default to calm
                            selected_emotion = current_agent_emotion if current_agent_emotion else "calm"
                            logging.warning(f"    [NO EMOTION TAG] Agent LLM didn't provide tag, using '{selected_emotion}'")

                        # Track emotion shift
                        if current_agent_emotion and current_agent_emotion != selected_emotion:
                            logging.info(f"    [EMOTION SHIFT] Agent: {current_agent_emotion} → {selected_emotion}")
                        current_agent_emotion = selected_emotion

                        # Build final utterance with validated emotion tag
                        agent_para_tag_str = f"(emotion:{selected_emotion})"
                        agent_utterance = f"{agent_para_tag_str} {clean_utterance}"

                        dialogue_turns.append(f"Agent: {agent_utterance}")
                        # Include emotion tag in history so LLM can see emotional progression
                        conversation_history_with_tags.append(f"Agent: {agent_utterance}")

                        logging.info(f"    [AGENT EMOTION] Turn {turn}: {selected_emotion}")
                    else:
                        logging.error(f"Could not generate agent response after {max_retries} retries, ending dialogue early")
                        break

            # Save dialogue (tags embedded in parentheses)
            if dialogue_turns:
                dialogue_id = f"{scenario['id']}_{dialogue_idx + 1}"
                save_dialogue(out_dir, dialogue_id, dialogue_turns)

                logging.info(f"Saved dialogue: {dialogue_id}")
                print(f"\n=== Generated Dynamic Emotion Dialogue {dialogue_idx + 1} for {scenario['id']} ===")
                for i, turn_text in enumerate(dialogue_turns[:6]):
                    print(turn_text)
                if len(dialogue_turns) > 6:
                    print("...")
                print("=" * 60)

# ────────────────────────────  POST-PROCESSING  ─────────────────────────────

def analyze_paralinguistic_distribution(cfg):
    """
    Analyze the distribution of paralinguistic attributes across all generated dialogues.
    For ADAPTATION MODE, analyzes both user and agent para tags.
    """
    src_dir = Path(cfg.dialogue["out_dir"])
    report_path = src_dir / "paralinguistic_analysis_adaptation.json"

    # Load paralinguistic pools from config
    paralinguistic_pools = cfg.get("paralinguistic_pools", {})

    # Initialize counters for both user and agent
    user_distribution = {dimension: {} for dimension in paralinguistic_pools.keys()}
    agent_distribution = {dimension: {} for dimension in paralinguistic_pools.keys()}

    total_user_turns = 0
    total_agent_turns = 0
    user_turns_with_tags = 0
    agent_turns_with_tags = 0

    # Process all dialogue .txt files
    for dialogue_file in src_dir.glob("*.txt"):
        with open(dialogue_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("User:"):
                total_user_turns += 1
                user_text = line.replace("User:", "").strip()
                clean_text, para_tags = extract_paralinguistic_tags(user_text)

                if para_tags:
                    user_turns_with_tags += 1
                    for dimension, value in para_tags.items():
                        if dimension in user_distribution:
                            if value in user_distribution[dimension]:
                                user_distribution[dimension][value] += 1
                            else:
                                user_distribution[dimension][value] = 1

            elif line.startswith("Agent:"):
                total_agent_turns += 1
                agent_text = line.replace("Agent:", "").strip()
                clean_text, para_tags = extract_paralinguistic_tags(agent_text)

                if para_tags:
                    agent_turns_with_tags += 1
                    for dimension, value in para_tags.items():
                        if dimension in agent_distribution:
                            if value in agent_distribution[dimension]:
                                agent_distribution[dimension][value] += 1
                            else:
                                agent_distribution[dimension][value] = 1

    # Create report
    report = {
        "mode": "ADAPTATION",
        "user_stats": {
            "total_turns": total_user_turns,
            "turns_with_tags": user_turns_with_tags,
            "coverage": user_turns_with_tags / total_user_turns if total_user_turns > 0 else 0,
            "distribution": user_distribution
        },
        "agent_stats": {
            "total_turns": total_agent_turns,
            "turns_with_tags": agent_turns_with_tags,
            "coverage": agent_turns_with_tags / total_agent_turns if total_agent_turns > 0 else 0,
            "distribution": agent_distribution
        }
    }

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(f"Paralinguistic analysis saved to: {report_path}")
    print(f"\n=== Paralinguistic Analysis (ADAPTATION MODE) ===")
    print(f"USER - Total Turns: {total_user_turns}, Turns with Tags: {user_turns_with_tags}, Coverage: {report['user_stats']['coverage']:.2%}")
    print(f"AGENT - Total Turns: {total_agent_turns}, Turns with Tags: {agent_turns_with_tags}, Coverage: {report['agent_stats']['coverage']:.2%}")
    print("=" * 60)

# ────────────────────────────────  TTS  ────────────────────────────────────
# TTS functions are identical to syn_para.py - reuse all TTS code
# (IndexTTS_gen, tts_batch, etc.)

# Import all TTS functions from syn_para.py to avoid code duplication
# These functions handle tag extraction and TTS generation identically
from syn_para import (
    detect_speaker_gender,
    pre_detect_first_gender_tag,
    get_opposite_gender,
    apply_audio_post_processing,
    get_emotion_reference_audio,
    IndexTTS_gen,
    tts_batch
)

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
        "out_file": "data_adaptation/scenarios.json",
        "prompt": "Generate {n} diverse customer service scenarios in JSON format."
    })

    # dialogue (ADAPTATION MODE)
    dialogue: OmegaConf = OmegaConf.create({
        "user_model": "meta-llama/llama-3.3-70b-instruct",
        "agent_model": "openai/gpt-4o-mini",
        "user_prompt": "You are a customer/user interacting with a support agent. Be natural and conversational.",
        "agent_prompt": "You are a helpful customer support agent. Adapt your tone and response based on the user's emotional state and characteristics.",
        "out_dir": "data_adaptation/dialog_txt",
        "min_turns": 4,
        "max_turns": 10,
        "per_scenario": 3,
        "user_para_probability": 0.8,  # 80% of user turns will have paralinguistic tags
        "user_para_dimension": "emotion",  # Fixed dimension for user (can be: emotion, age, gender, sarcasm)
    })

    # paralinguistic_pools (loaded from config file)
    paralinguistic_pools: OmegaConf = OmegaConf.create({})

    # tts (IndexTTS2)
    tts: OmegaConf = OmegaConf.create({
        "wav_dir": "data_adaptation/tts_audio",
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
            print("Dialogue Generating (ADAPTATION MODE)...")
            generate_adaptation_dialogues(self.cfg)

        if "analysis" in st:
            print("Analyzing Paralinguistic Distribution (ADAPTATION MODE)...")
            analyze_paralinguistic_distribution(self.cfg)

        if "tts" in st:
            print("Generating TTS Audio (IndexTTS2)...")
            tts_batch(self.cfg)

# ─────────────────────────────  CLI ENTRY  ────────────────────────────────

def main():  # pragma: no cover
    @hydra.main(config_path="conf", config_name="base_adaptation", version_base=None)
    def _run(cfg):
        if not isinstance(cfg, PipelineConfig):
            cfg = OmegaConf.merge(OmegaConf.structured(PipelineConfig), cfg)

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        Pipeline(cfg).run()

    print("Finish Config Matching")
    _run()

if __name__ == "__main__":
    main()

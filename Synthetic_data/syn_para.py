from __future__ import annotations

import json
import re
import os
import random
import logging
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
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

def generate_paralinguistic_tag(
    pools: Dict[str, List[str]],
    mode: str = "single",
    multi_para_config: Dict = None,
    control_config: Dict = None,
    category_history: Dict[str, str] = None
) -> Dict[str, str]:
    """
    Unified function for generating paralinguistic tags with different modes.

    Merges the logic of generate_single_paralinguistic_tag, generate_multi_paralinguistic_tags,
    and select_control_target into one flexible function.

    Args:
        pools: Dictionary of paralinguistic dimension pools from config
        mode: Generation strategy - "single", "multi", or "control"
            - "single": Randomly pick one dimension and one value
            - "multi": Pick multiple dimensions with probabilistic cascading (for multi-para control)
            - "control": Pick fixed dimension, excluding current value (for single-para control)
        multi_para_config: Required for mode="multi". Dict with:
            - allowed_dimensions: List[str] - dimensions that can be combined
            - add_probability: float (default 0.5) - probability of adding subsequent tags
            - max_tags: int (default 3) - maximum number of tags to generate
        control_config: Required for mode="control". Dict with:
            - fixed_dimension: str - the specific dimension to use
        category_history: Optional dict of per-category last used values {dimension: value}
                         Used across ALL modes for consecutive prevention to avoid immediate repetition

    Returns:
        Dictionary with paralinguistic tag(s)
        e.g., {"speed": "fast"} for single/control, or {"speed": "fast", "pitch": "high"} for multi
    """
    if mode == "single":
        # ─────────────────────────────────────────────────────────────
        # SINGLE MODE: Randomly pick one dimension and value
        # ─────────────────────────────────────────────────────────────
        dimension = random.choice(list(pools.keys()))
        value = random.choice(pools[dimension])
        return {dimension: value}

    elif mode == "multi":
        # ─────────────────────────────────────────────────────────────
        # MULTI MODE: Pick multiple dimensions with probabilistic cascading
        # Used for multi-para control requests
        # Excludes values from category_history to prevent consecutive repetition
        # ─────────────────────────────────────────────────────────────
        if not multi_para_config:
            raise ValueError("multi_para_config required for mode='multi'")

        allowed_dimensions = multi_para_config.get("allowed_dimensions", [])
        add_probability = multi_para_config.get("add_probability", 0.5)
        max_tags = multi_para_config.get("max_tags", 3)

        # Filter pools to only allowed dimensions
        available_dims = [d for d in allowed_dimensions if d in pools]

        if not available_dims:
            logging.warning("No available dimensions for multi-para mode")
            return {}

        # Shuffle to randomize order
        random.shuffle(available_dims)

        # Helper function to select value excluding from history (consecutive prevention)
        def select_value_excluding_history(dim):
            available_values = list(pools[dim])
            # Exclude value from category_history to prevent consecutive repetition
            if category_history and dim in category_history:
                history_val = category_history[dim]
                available_values = [v for v in available_values if v != history_val]
                if available_values:
                    logging.info(f"    [CATEGORY-HISTORY] Excluding '{history_val}' for {dim} (consecutive prevention)")
                else:
                    # All values filtered out, use full pool
                    available_values = list(pools[dim])
                    logging.warning(f"    [CATEGORY-HISTORY] All values filtered for {dim}, using full pool")
            return random.choice(available_values)

        # Start with one mandatory tag
        result = {}
        first_dim = available_dims[0]
        result[first_dim] = select_value_excluding_history(first_dim)
        remaining_dims = available_dims[1:]

        # Probabilistically add more tags
        for dim in remaining_dims:
            if len(result) >= max_tags:
                break
            if random.random() < add_probability:
                result[dim] = select_value_excluding_history(dim)
            else:
                # Once we fail the probability check, stop adding more
                break

        logging.info(f"    [MULTI-PARA] Generated {len(result)} tags: {result}")
        return result

    elif mode == "control":
        # ─────────────────────────────────────────────────────────────
        # CONTROL MODE: Pick fixed dimension, excluding value from history
        # Used for single-para control requests (select_control_target logic)
        # Uses category_history for consecutive prevention (same as multi mode)
        # ─────────────────────────────────────────────────────────────
        if not control_config:
            raise ValueError("control_config required for mode='control'")

        fixed_dimension = control_config.get("fixed_dimension")

        if not fixed_dimension:
            raise ValueError("fixed_dimension required in control_config")

        dimension = fixed_dimension
        available_values = list(pools[dimension])  # Make a copy

        # Exclude value from category_history (consecutive prevention)
        if category_history and dimension in category_history:
            history_val = category_history[dimension]
            available_values = [v for v in available_values if v != history_val]
            if available_values:
                logging.info(f"    [CATEGORY-HISTORY] Excluding '{history_val}' for {dimension} (consecutive prevention)")
            else:
                # All values filtered out, fall back to original pool (edge case)
                logging.warning(f"    [CATEGORY-HISTORY] All values filtered out, using full pool")
                available_values = list(pools[dimension])

        target_value = random.choice(available_values)
        return {dimension: target_value}

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'single', 'multi', or 'control'")


def format_multi_paralinguistic_tags(tags: Dict[str, str]) -> str:
    """
    Format multiple paralinguistic tags into parentheses format.
    Format: (speed:fast, pitch:high, emotion:happy)

    Args:
        tags: Dictionary of dimension:value pairs

    Returns: Formatted string like "(speed:fast, pitch:high)"
    """
    if not tags:
        return ""
    tag_str = ", ".join([f"{k}:{v}" for k, v in tags.items()])
    return f"({tag_str})"


def count_paralinguistic_tags(text: str) -> int:
    """
    Count the number of paralinguistic tags in a text string.

    Args:
        text: Text that may start with (tag1:val1, tag2:val2, ...)

    Returns: Number of tags found (0 if no tags)
    """
    if not text.startswith("("):
        return 0

    closing = text.find(")")
    if closing == -1:
        return 0

    tag_string = text[1:closing].strip()
    if not tag_string:
        return 0

    # Count comma-separated pairs
    pairs = [p.strip() for p in tag_string.split(',') if ':' in p]
    return len(pairs)

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

def generate_scenarios(cfg, topic):
    """Generate scenarios for a single topic."""
    out_path = Path(cfg.data_root) / "scenarios" / topic / "scenarios.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = cfg.scenario["prompt"].format(n=cfg.scenario["n"], topic=topic)
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


# ─────────────────────────────  PARALINGUISTIC DIALOGUE GEN  ──────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR DIALOGUE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _determine_control_mode(
    turn_pair_idx: int,
    has_had_control_request: bool,
    control_request_frequency_first: float,
    control_request_frequency_subsequent: float
) -> bool:
    """
    Determine if this turn pair should be in control mode.

    Args:
        turn_pair_idx: Index of the current turn pair (0-based)
        has_had_control_request: Whether dialogue has had any control request yet
        control_request_frequency_first: Probability for first control request
        control_request_frequency_subsequent: Probability for subsequent requests

    Returns:
        True if this should be a control mode turn pair
    """
    # Use two-tier probability system
    if has_had_control_request:
        # Already had control - use lower probability
        return random.random() < control_request_frequency_subsequent
    else:
        # First control request - use higher probability
        return random.random() < control_request_frequency_first


def _generate_control_para_tags(
    multi_para_mode: bool,
    multi_para_config: Dict,
    control_dimension: str,
    category_history: Optional[Dict[str, str]],
    paralinguistic_pools: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Generate paralinguistic tags for control mode.

    Args:
        multi_para_mode: Whether to use multi-para mode
        multi_para_config: Configuration for multi-para mode
        control_dimension: Fixed dimension for single-para mode
        category_history: Per-category last used values {dimension: value}
        paralinguistic_pools: Available paralinguistic options

    Returns:
        Dictionary of para tags {dimension: value}
    """
    if multi_para_mode:
        # Multi-para mode: generate multiple NEW tags, then preserve unchanged dimensions
        new_tags = generate_paralinguistic_tag(
            pools=paralinguistic_pools,
            mode="multi",
            multi_para_config=multi_para_config,
            category_history=category_history  # Use history for consecutive prevention
        )

        # PRESERVE unchanged dimensions from category_history
        if category_history:
            preserved_tags = {}
            for dim, value in category_history.items():
                if dim not in new_tags:  # Dimension not being changed
                    preserved_tags[dim] = value
                    logging.info(f"    [PRESERVE] Keeping {dim}:{value} from history (unchanged)")

            # Merge: preserved_tags first, then new_tags overwrite
            result = {**preserved_tags, **new_tags}
            logging.info(f"    [MULTI-PARA RESULT] Final tags after preservation: {result}")
            return result

        return new_tags
    else:
        # Single-para mode: generate one tag using category_history for consecutive prevention
        return generate_paralinguistic_tag(
            pools=paralinguistic_pools,
            mode="control",
            control_config={
                "fixed_dimension": control_dimension
            },
            category_history=category_history  # Use history for consecutive prevention
        )


def _build_user_instruction(
    is_first_turn: bool,
    is_control_mode: bool,
    control_para_tags: Optional[Dict[str, str]],
    context_info: str,
    conversation_history: List[str],
    multi_para_mode: bool,
    category_history: Optional[Dict[str, str]] = None
) -> str:
    """
    Build instruction prompt for user LLM.

    Args:
        is_first_turn: Whether this is the first turn (turn 0)
        is_control_mode: Whether this is a control mode turn
        control_para_tags: Para tags to request (if control mode)
        context_info: Scenario context
        conversation_history: Previous conversation turns
        multi_para_mode: Whether multi-para mode is enabled
        category_history: The agent's CURRENT para tags (so LLM knows what to change FROM)

    Returns:
        Instruction string for user LLM
    """
    # First turn WITHOUT control mode - start normal conversation
    if is_first_turn and not is_control_mode:
        return (
            f"{context_info}"
            f"This is the start of the conversation. "
            f"Generate the user's first message based on the scenario. "
            f"Be natural and conversational. Only output the user's utterance, nothing else."
        )

    history_text = "\n".join(conversation_history) if conversation_history else ""

    # First turn WITH control mode - start with voice control request
    # IMPORTANT: First turn cannot use "switch" or "change" language since there's no previous voice state
    if is_first_turn and is_control_mode and control_para_tags:
        if multi_para_mode and len(control_para_tags) > 1:
            # Multi-para: request multiple characteristics as opening
            tag_requests = ", ".join([f"{k}={v}" for k, v in control_para_tags.items()])
            tag_list = "\n".join([f"- {k}: {v}" for k, v in control_para_tags.items()])
            return (
                f"{context_info}"
                f"This is the start of the conversation. "
                f"Generate the user's FIRST message where they immediately ask the agent to speak in a specific way.\n"
                f"Ask them to use MULTIPLE voice characteristics:\n{tag_list}\n\n"
                f"Be creative and natural - combine these requests naturally as an opening.\n"
                f"IMPORTANT: Do NOT use words like 'switch' or 'change' since this is the first interaction.\n"
                f"Instead, simply ask them to 'speak with' or 'use' these characteristics.\n"
                f"Generate a DIVERSE, NATURAL opening request for: {tag_requests}\n"
                f"Then state your question/need based on the scenario.\n"
                f"Only output the user's utterance, nothing else."
            )
        else:
            # Single-para: request one characteristic as opening
            ctrl_dim, ctrl_value = next(iter(control_para_tags.items()))
            return (
                f"{context_info}"
                f"This is the start of the conversation. "
                f"Generate the user's FIRST message where they immediately ask the agent to speak in a specific way.\n"
                f"Specifically, ask them to speak with {ctrl_dim} as '{ctrl_value}'.\n"
                f"Be creative and natural - use your own words, don't use a template.\n"
                f"IMPORTANT: Do NOT use words like 'switch' or 'change' since this is the first interaction.\n"
                f"Instead, simply ask them to 'speak with' or 'use' {ctrl_dim}='{ctrl_value}'.\n"
                f"Generate a DIVERSE, NATURAL opening request for {ctrl_dim}='{ctrl_value}'.\n"
                f"Then state your question/need based on the scenario.\n"
                f"Only output the user's utterance, nothing else."
            )

    if is_control_mode and control_para_tags:
        # Control mode: user requests voice change
        if multi_para_mode and len(control_para_tags) > 1:
            # Multi-para: request multiple changes
            tag_requests = ", ".join([f"{k}={v}" for k, v in control_para_tags.items()])
            tag_list = "\n".join([f"- {k}: {v}" for k, v in control_para_tags.items()])

            # Build current state info for LLM context
            current_state_info = ""
            if category_history:
                current_tags_str = ", ".join([f"{k}='{v}'" for k, v in category_history.items()])
                current_state_info = f"The agent is currently speaking with: {current_tags_str}\n"

            return (
                f"{context_info}"
                f"Conversation so far:\n{history_text}\n\n"
                f"Generate the user's next response where they ask the agent to change their voice.\n"
                f"{current_state_info}"
                f"Ask them to adjust MULTIPLE voice characteristics at once:\n{tag_list}\n\n"
                f"Be creative and natural - combine these requests naturally.\n"
                f"Generate a DIVERSE, NATURAL request for: {tag_requests}\n"
                f"Then optionally continue with the conversation topic.\n"
                f"Only output the user's utterance, nothing else."
            )
        else:
            # Single-para: request one change
            ctrl_dim, ctrl_value = next(iter(control_para_tags.items()))

            # Build current state info so LLM knows what to change FROM
            current_state_info = ""
            if category_history and ctrl_dim in category_history:
                current_val = category_history[ctrl_dim]
                current_state_info = f"The agent is currently speaking with {ctrl_dim}='{current_val}'.\n"

            return (
                f"{context_info}"
                f"Conversation so far:\n{history_text}\n\n"
                f"Generate the user's next response where they ask the agent to change their voice.\n"
                f"{current_state_info}"
                f"Specifically, ask them to change their {ctrl_dim} to '{ctrl_value}'.\n"
                f"Be creative and natural - use your own words, don't use a template.\n"
                f"Generate a DIVERSE, NATURAL request for {ctrl_dim}='{ctrl_value}'.\n"
                f"Then optionally continue with the conversation topic.\n"
                f"Only output the user's utterance, nothing else."
            )
    else:
        # Normal mode: regular conversation
        return (
            f"{context_info}"
            f"Conversation so far:\n{history_text}\n\n"
            f"Generate the user's next response based on the agent's last message. "
            f"Be natural and conversational. Only output the user's utterance, nothing else."
        )


def _build_agent_instruction_and_tags(
    is_control_mode: bool,
    control_para_tags: Optional[Dict[str, str]],
    category_history: Optional[Dict[str, str]],
    context_info: str,
    conversation_history: List[str],
    multi_para_mode: bool,
    paralinguistic_pools: Dict[str, List[str]]
) -> Tuple[str, Optional[Dict[str, str]], bool]:
    """
    Build instruction prompt for agent LLM and determine para tags to apply.

    Args:
        is_control_mode: Whether this is a control mode turn
        control_para_tags: Para tags requested by user
        category_history: Last emotion tags used by agent (full dict)
        context_info: Scenario context
        conversation_history: Previous conversation turns
        multi_para_mode: Whether multi-para mode is enabled
        paralinguistic_pools: Available paralinguistic options

    Returns:
        Tuple of (instruction, para_tags_to_apply, use_control_prompt)
    """
    history_text = "\n".join(conversation_history)

    if is_control_mode and control_para_tags:
        # Control mode: apply requested para tags
        para_tags = control_para_tags

        if multi_para_mode and len(control_para_tags) > 1:
            # Multi-para control
            para_tag_str = format_multi_paralinguistic_tags(para_tags)
            tag_list = ", ".join([f"{k} to '{v}'" for k, v in para_tags.items()])

            para_instruction = (
                f"\n\nIMPORTANT: The user just asked you to change MULTIPLE voice characteristics:\n"
                f"{tag_list}\n\n"
                f"You should:\n"
                f"1. Acknowledge the requests naturally \n"
                f"2. Then continue helping with their question/topic\n\n"
                f"Your response MUST start with ALL the tags in parentheses EXACTLY as shown:\n"
                f"{para_tag_str}\n\n"
                f"Format: {para_tag_str} [brief acknowledgment] [continue conversation]\n"
            )
        else:
            # Single-para control
            ctrl_dim, ctrl_value = next(iter(control_para_tags.items()))
            para_tag_str = format_paralinguistic_tags(para_tags)

            para_instruction = (
                f"\n\nIMPORTANT: The user just asked you to change your {ctrl_dim} to '{ctrl_value}'.\n"
                f"You should:\n"
                f"1. Acknowledge the request naturally\n"
                f"2. Then continue helping with their question/topic\n\n"
                f"Your response MUST start with the tag in parentheses EXACTLY as shown:\n"
                f"{para_tag_str}\n\n"
                f"Format: {para_tag_str} [brief acknowledgment] [continue conversation]\n"
            )

        agent_instruction = (
            f"{context_info}"
            f"Conversation so far:\n{history_text}\n\n"
            f"Generate the agent's response to the user's last message. "
            f"Be helpful, professional, and natural. "
            f"{para_instruction}"
            f"Only output the agent's utterance (with the paralinguistic tag if specified), nothing else."
        )

        return agent_instruction, para_tags, True  # Use control prompt

    elif category_history:
        # Normal mode with emotion continuity - maintain ALL previous tags
        para_tags = category_history.copy()

        # Format tags for display
        if len(para_tags) > 1:
            para_tag_str = format_multi_paralinguistic_tags(para_tags)
            tag_list = ", ".join([f"{k}='{v}'" for k, v in para_tags.items()])
            para_instruction = (
                f"\n\nIMPORTANT: For emotional continuity, maintain ALL the same voice characteristics as your last response:\n"
                f"{tag_list}\n\n"
                f"Your response MUST start with ALL tags in parentheses EXACTLY as shown:\n"
                f"{para_tag_str}\n\n"
                f"Format: {para_tag_str} [your response]\n"
                f"This ensures your voice characteristics remain consistent throughout the conversation.\n"
            )
        else:
            para_tag_str = format_paralinguistic_tags(para_tags)
            ctrl_dim, ctrl_value = next(iter(para_tags.items()))
            para_instruction = (
                f"\n\nIMPORTANT: For emotional continuity, maintain the same {ctrl_dim} as your last response ('{ctrl_value}').\n"
                f"Your response MUST start with the tag in parentheses EXACTLY as shown:\n"
                f"{para_tag_str}\n\n"
                f"Format: {para_tag_str} [your response]\n"
                f"This ensures your voice characteristics remain consistent throughout the conversation.\n"
            )

        agent_instruction = (
            f"{context_info}"
            f"Conversation so far:\n{history_text}\n\n"
            f"Generate the agent's response to the user's last message. "
            f"Be helpful, professional, and natural. "
            f"{para_instruction}"
            f"Only output the agent's utterance (with the paralinguistic tag(s) if specified), nothing else."
        )

        return agent_instruction, para_tags, True  # Use control prompt

    else:
        # Normal mode without emotion history - NO paralinguistic tags
        # Agent only gets para tags when user explicitly requests them via control mode
        logging.info(f"    [NORMAL] No emotion history - agent will speak WITHOUT paralinguistic tags")

        para_tags = None

        agent_instruction = (
            f"{context_info}"
            f"Conversation so far:\n{history_text}\n\n"
            f"Generate the agent's response to the user's last message. "
            f"Be helpful, professional, and natural. "
            f"Only output the agent's utterance, nothing else."
        )

        return agent_instruction, para_tags, False  # Use normal prompt (no para tags)


def _call_llm_with_retry(
    model: str,
    system_prompt: str,
    user_instruction: str,
    role_prefix: str,
    turn_idx: int,
    max_retries: int = 3
) -> Optional[str]:
    """
    Call LLM with retry logic.

    Args:
        model: Model name
        system_prompt: System prompt for LLM
        user_instruction: User instruction for LLM
        role_prefix: Prefix to strip from response (e.g., "User:", "Agent:")
        turn_idx: Turn index for logging
        max_retries: Maximum number of retries

    Returns:
        Generated utterance, or None if all retries failed
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction}
    ]

    for retry in range(max_retries):
        try:
            utterance = chat_completion(
                model,
                messages,
                max_tokens=512,
                temperature=0.9,
                top_p=0.9
            ).strip()

            # Clean up role prefix
            utterance = re.sub(f'^{role_prefix}\\s*', '', utterance, flags=re.IGNORECASE)

            if utterance:
                return utterance
            else:
                logging.warning(f"Empty response from LLM at turn {turn_idx}, retry {retry + 1}/{max_retries}")

        except Exception as e:
            logging.error(f"Error in LLM turn {turn_idx} (retry {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                wait_time = 2 ** retry
                logging.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed after {max_retries} retries")

    return None


def generate_paralinguistic_dialogues(cfg, topic, mode):
    """
    PARALINGUISTIC VERSION: Generates dialogues with paralinguistic control tags.

    REFACTORED TO PROCESS TURN PAIRS (user+agent) TOGETHER.

    MODES:
    - CONTROL MODE: User requests voice change, agent applies it (multi-para or single para)
    - NORMAL MODE: User speaks normally, agent maintains emotion continuity if exists

    Key features:
    1. Processes user+agent turn pairs together to reduce nesting
    2. Control mode can start from turn 0 (no restriction)
    3. Cleaner separation of control vs normal mode logic
    4. Helper functions extract complex logic
    """
    # Read scenarios
    scen_path = Path(cfg.data_root) / "scenarios" / topic / "scenarios.jsonl"
    out_dir = Path(cfg.data_root) / mode / topic / "dialogue_txt"
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
    control_request_frequency_first = cfg.dialogue.get("control_request_frequency_first", 0.8)
    control_request_frequency_subsequent = cfg.dialogue.get("control_request_frequency_subsequent", 0.4)

    # Derive multi_para_mode and control_dimension from the mode parameter
    if mode == "multi":
        multi_para_mode = True
        control_dimension = cfg.dialogue.get("control_dimension", "speed")  # unused in multi mode
    else:
        multi_para_mode = False
        control_dimension = mode  # e.g. "gender", "speed", "emotion", ...

    # Multi-para mode configuration
    multi_para_probability = cfg.dialogue.get("multi_para_probability", 0.7)
    max_para_tags = cfg.dialogue.get("max_para_tags", 3)
    multi_para_dimensions = cfg.dialogue.get("multi_para_dimensions", ["speed", "pitch", "emotion", "volume", "age"])

    # Multi-para config for helper functions
    multi_para_config = {
        "allowed_dimensions": multi_para_dimensions,
        "add_probability": multi_para_probability,
        "max_tags": max_para_tags
    }

    # Load paralinguistic pools from config
    paralinguistic_pools = cfg.get("paralinguistic_pools", {})

    # mode_suffix matches mode param ("multi" or dimension name)
    mode_suffix = mode

    logging.info(f"Generating PARALINGUISTIC dialogues:")
    logging.info(f"  Topic: {topic}")
    logging.info(f"  Mode: {mode_suffix}")
    logging.info(f"  User Model: {user_model}")
    logging.info(f"  Agent Model: {agent_model}")
    logging.info(f"  Turn Range: {min_turns}-{max_turns}")
    logging.info(f"  Dialogues per Scenario: {dialogues_per_scenario}")
    logging.info(f"  Control Request Frequency (First): {control_request_frequency_first}")
    logging.info(f"  Control Request Frequency (Subsequent): {control_request_frequency_subsequent}")
    if multi_para_mode:
        logging.info(f"  ═══ MULTI-PARA MODE ENABLED ═══")
        logging.info(f"  Multi-Para Probability: {multi_para_probability}")
        logging.info(f"  Max Para Tags: {max_para_tags}")
        logging.info(f"  Multi-Para Dimensions: {multi_para_dimensions}")
    else:
        logging.info(f"  Control Dimension (Fixed): {control_dimension}")

    for scenario in tqdm(scenarios, desc="dialogues"):
        scenario_desc = json.dumps(scenario, ensure_ascii=False)

        # Extract scenario index from scenario['id'] (e.g., "scenario1" -> "1")
        scenario_idx = scenario['id'].replace('scenario', '')

        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_turns = []
            conversation_history = []

            # Track if dialogue has had any control request yet
            has_had_control_request = False

            # Track per-category history for consecutive prevention and tag preservation
            # Stores the last value used for each paralinguistic dimension {dimension: value}
            # Used for: (1) consecutive prevention, (2) tag preservation in multi-para, (3) emotion continuity
            category_history = {}  # Format: Dict[str, str] - {dimension: last_value}

            # ═══════════════════════════════════════════════════════════════
            # RESTRICT GENDER POOL PER DIALOGUE
            # ═══════════════════════════════════════════════════════════════
            # For each dialogue, randomly choose EITHER (man, normal) OR (woman, normal)
            # This prevents confusing gender switches within a single conversation
            dialogue_paralinguistic_pools = paralinguistic_pools.copy()
            if "gender" in paralinguistic_pools:
                # Randomly choose which gender pair to use for this dialogue
                use_male = random.choice([True, False])
                if use_male:
                    # Use (man, normal) only
                    dialogue_paralinguistic_pools["gender"] = ["male", "normal"]
                    logging.info(f"  -> Gender pool for this dialogue: (male, normal)")
                else:
                    # Use (woman, normal) only
                    dialogue_paralinguistic_pools["gender"] = ["female", "normal"]
                    logging.info(f"  -> Gender pool for this dialogue: (female, normal)")

            context_info = f"Scenario: {scenario_desc}\n\n"
            num_turns = random.randint(min_turns, max_turns)
            # Ensure num_turns is even (dialogues must end with agent response)
            if num_turns % 2 != 0:
                num_turns += 1
            logging.info(f"  Generating {scenario['id']}_{dialogue_idx + 1} with {num_turns} turns")

            # ═══════════════════════════════════════════════════════════════
            # MAIN LOOP: Process turn pairs (user + agent) together
            # ═══════════════════════════════════════════════════════════════
            num_turn_pairs = num_turns // 2

            for turn_pair_idx in range(num_turn_pairs):
                is_first_turn = (turn_pair_idx == 0)
                turn_idx_user = turn_pair_idx * 2
                turn_idx_agent = turn_pair_idx * 2 + 1

                # ─────────────────────────────────────────────────────────────
                # STEP 1: Determine mode for this turn pair
                # ─────────────────────────────────────────────────────────────
                is_control_mode = _determine_control_mode(
                    turn_pair_idx,
                    has_had_control_request,
                    control_request_frequency_first,
                    control_request_frequency_subsequent
                )

                # Generate control para tags if in control mode
                control_para_tags = None
                if is_control_mode:
                    control_para_tags = _generate_control_para_tags(
                        multi_para_mode,
                        multi_para_config,
                        control_dimension,
                        category_history,
                        dialogue_paralinguistic_pools  # Use dialogue-specific pool
                    )
                    has_had_control_request = True

                    if multi_para_mode and len(control_para_tags) > 1:
                        logging.info(f"    [CONTROL] Turn {turn_idx_user}: Multi-para mode - {control_para_tags}")
                    else:
                        ctrl_dim, ctrl_value = next(iter(control_para_tags.items()))
                        logging.info(f"    [CONTROL] Turn {turn_idx_user}: Single-para mode - {ctrl_dim}={ctrl_value}")

                # ─────────────────────────────────────────────────────────────
                # STEP 2: Generate User utterance
                # ─────────────────────────────────────────────────────────────
                user_instruction = _build_user_instruction(
                    is_first_turn,
                    is_control_mode,
                    control_para_tags,
                    context_info,
                    conversation_history,
                    multi_para_mode,
                    category_history=category_history  # Pass current para tags so LLM knows what to change FROM
                )

                user_utterance = _call_llm_with_retry(
                    user_model,
                    control_user_prompt,
                    user_instruction,
                    "User:",
                    turn_idx_user
                )

                if not user_utterance:
                    logging.error(f"Could not generate user response, ending dialogue early")
                    break

                # Add to dialogue
                dialogue_turns.append(f"User: {user_utterance}")
                conversation_history.append(f"User: {user_utterance}")

                # ─────────────────────────────────────────────────────────────
                # STEP 3: Generate Agent utterance
                # ─────────────────────────────────────────────────────────────
                agent_instruction, agent_para_tags, use_control_prompt = _build_agent_instruction_and_tags(
                    is_control_mode,
                    control_para_tags,
                    category_history,
                    context_info,
                    conversation_history,
                    multi_para_mode,
                    dialogue_paralinguistic_pools  # Use dialogue-specific pool
                )

                # Choose system prompt
                agent_sys_prompt = control_agent_prompt if use_control_prompt else agent_system_prompt

                agent_utterance = _call_llm_with_retry(
                    agent_model,
                    agent_sys_prompt,
                    agent_instruction,
                    "Agent:",
                    turn_idx_agent
                )

                if not agent_utterance:
                    logging.error(f"Could not generate agent response, ending dialogue early")
                    break

                # Ensure paralinguistic tag is present if expected
                if agent_para_tags and not agent_utterance.startswith("("):
                    para_tag_str = format_paralinguistic_tags(agent_para_tags)
                    agent_utterance = f"{para_tag_str} {agent_utterance}"

                # Add to dialogue
                dialogue_turns.append(f"Agent: {agent_utterance}")

                # Extract tags and update tracking
                clean_utterance, extracted_tags = extract_paralinguistic_tags(agent_utterance)

                # IMPORTANT: Only store tags if we were EXPECTING them (agent_para_tags is not None)
                # This prevents the LLM from accidentally initializing tags before control mode happens
                if extracted_tags and agent_para_tags is not None:
                    # Update category history with ALL extracted tags (accumulative per-category tracking)
                    category_history.update(extracted_tags)
                    tag_summary = ", ".join([f"{k}={v}" for k, v in extracted_tags.items()])
                    logging.info(f"    [CATEGORY-HISTORY UPDATE] Current history: {category_history}")
                elif extracted_tags and agent_para_tags is None:
                    # LLM generated tags when it shouldn't have - log warning
                    logging.warning(f"    [WARNING] LLM generated unexpected tags {extracted_tags} - ignoring them")

                # Add clean version to conversation history
                conversation_history.append(f"Agent: {clean_utterance}")

            # Save dialogue (tags embedded in parentheses)
            if dialogue_turns:
                # New format: {topic}_{control_dimension or "multi"}_{scenario_idx}_{dialogue_idx}
                # Note: dialogue_idx is needed for unique filenames
                dialogue_id = f"{topic}_{mode_suffix}_{scenario_idx}_{dialogue_idx + 1}"
                save_dialogue(out_dir, dialogue_id, dialogue_turns)

                logging.info(f"Saved dialogue: {dialogue_id}")
                print(f"\n=== Generated Paralinguistic Dialogue {dialogue_idx + 1} for {scenario['id']} ===")
                print("\n".join(dialogue_turns[:5]))  # Print first 5 turns
                if len(dialogue_turns) > 5:
                    print("...")
                print("=" * 60)

# ────────────────────────────────  TTS - IndexTTS2  ────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# TTS HELPER FUNCTIONS - Refactored for readability
# ═══════════════════════════════════════════════════════════════════════════

def _detect_speaker_gender(spk_audio_path, spk_audio_base_dir):
    """
    Detect the gender of a speaker by checking if the audio file exists in male or female subdirectories.

    Args:
        spk_audio_path: Path to the speaker audio file
        spk_audio_base_dir: Base directory containing male/ and female/ subdirectories

    Returns:
        "male" or "female" or None if not found
    """
    spk_filename = Path(spk_audio_path).name

    # Check if file ei txists in male directory
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

def _pre_detect_first_gender_tag(script):
    """
    Pre-scan the script to find the first gender tag in agent turns.
    This is used to determine the initial speaker gender (opposite of first switch).
    Skips "normal" tags and looks for actual gender switches (male/female).

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
                # Skip "normal" - continue searching for actual gender switch
                if first_gender.lower() != "normal":
                    logging.info(f"  -> First gender tag detected: {first_gender}")
                    return first_gender

    logging.info(f"  -> No gender tags found in script")
    return None

def _get_opposite_gender(gender_value):
    """
    Get the opposite gender for initial speaker setup.

    Args:
        gender_value: Gender value from tag (e.g., "woman", "man", "female", "male")

    Returns:
        Opposite gender directory name ("male" or "female")
    """
    # Map to standard names
    gender_map = {
        "female": "female",
        "male": "male"
    }

    target_gender = gender_map.get(gender_value.lower())

    if target_gender == "female":
        return "male"  # If switching to female, start with male
    elif target_gender == "male":
        return "female"  # If switching to male, start with female
    else:
        return None

def _apply_audio_post_processing(audio_tensor, sample_rate, para_tags):
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
                # Save current audio to temporary input file using soundfile (better compatibility)
                # AudioStretch needs standard PCM format
                # Ensure audio_np is in correct shape: (channels, samples) -> (samples, channels) for soundfile
                if audio_np.shape[0] == 1:
                    # Mono: (1, samples) -> (samples,)
                    audio_for_save = audio_np[0]
                else:
                    # Stereo or multi-channel: (channels, samples) -> (samples, channels)
                    audio_for_save = audio_np.T

                # Save as 16-bit PCM WAV (most compatible format)
                sf.write(temp_input_path, audio_for_save, sample_rate, subtype='PCM_16')

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

                # Load the processed audio back using soundfile
                processed_audio_np, processed_sr = sf.read(temp_output_path, dtype='float32')

                # Ensure shape is (1, samples) for mono
                if processed_audio_np.ndim == 1:
                    # Mono: (samples,) -> (1, samples)
                    processed_audio_np = processed_audio_np[np.newaxis, :]
                else:
                    # Multi-channel: (samples, channels) -> (channels, samples)
                    processed_audio_np = processed_audio_np.T

                # Resample if necessary
                if processed_sr != sample_rate:
                    # Use librosa for resampling numpy arrays
                    processed_audio_np = librosa.resample(
                        processed_audio_np,
                        orig_sr=processed_sr,
                        target_sr=sample_rate
                    )

                # Update audio_np with processed result
                audio_np = processed_audio_np

                logging.info(f"     Audio shape after: {audio_np.shape}")

            finally:
                # Clean up temporary files
                import os
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                    
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

def _get_emotion_reference_audio(para_tags, emotion_audio_pool_dir, speaker_gender=None, emo_audio_cache=None):
    """
    Map paralinguistic tags to emotion reference audio files.
    Uses cache to reuse same audio for same tag within a dialogue.
    Uses IndexTTS-2's emotion reference audio feature (separate style prompt).

    Special handling for "normal" value:
    - When value is "normal", returns None to use speaker reference only
    - No emotion reference audio is applied
    - This allows natural voice without any style modification

    Special handling for gender mode (CROSS-GENDER STYLE TRANSFER):
    - Uses speaker_gender + target gender to select cross-gender reference
    - E.g., male speaker + (gender:female) → man_to_woman
    - E.g., female speaker + (gender:male) → woman_to_man
    - Speaker voice identity stays the same, only style changes

    Special handling for pitich mode:
    - Uses speaker_gender to select gender-appropriate pitch reference
    - E.g., "high" pitch + male speaker → high_pitch_man
    - E.g., "low" pitch + female speaker → low_pitch_woman

    Special handling for "very_" prefix:
    - Strips "very_" prefix when searching for directories
    - E.g., "very_fast" → searches "fast" directory
    - Returns alpha=1.0 for "very_*" tags, alpha=0.6 for normal tags

    Args:
        para_tags: Dict of paralinguistic tags (e.g., {"gender": "female"} or {"age": "elderly"})
        emotion_audio_pool_dir: Base directory containing emotion reference audio pool
        speaker_gender: "male" or "female" (required for gender mode and pitch mode)
        emo_audio_cache: Optional dict to cache selected audio for each tag.
                         Key: (dimension, value), Value: audio_path
                         Same tag reuses cached audio, different tag samples new audio.

    Returns:
        Tuple of (audio_path, emo_alpha) or (None, 1.0) if no tags or file not found
        - audio_path: Path to emotion reference audio file
        - emo_alpha: Emotion strength (1.0 for "very_*" or gender transfer, 0.6 for normal)

    Note:
        With emo_audio_cache: Same tag reuses cached audio (consistency within dialogue)
        Without cache: Randomly samples each time (variety across dialogues)

    Audio pool structure:
        emo/
        ├── man_to_woman/         # Gender: cross-gender style transfer
        │   ├── man_to_woman_0.mp3
        │   └── ...
        ├── woman_to_man/         # Gender: cross-gender style transfer
        │   ├── woman_to_man_0.mp3
        │   └── ...
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
        return None, 1.0

    # Get the dimension and value (should be only ONE per utterance)
    dimension, value = next(iter(para_tags.items()))

    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL HANDLING FOR "NORMAL" VALUE - USE SPEAKER REFERENCE ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    if value.lower() == "normal":
        logging.info(f"  ═══ NORMAL MODE ═══")
        logging.info(f"  -> Tag {dimension}=normal detected - using speaker reference ONLY (no emotion reference)")
        return None, 1.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK CACHE - Same tag reuses same audio within dialogue
    # ═══════════════════════════════════════════════════════════════════════════
    cache_key = (dimension, value)
    if emo_audio_cache is not None and cache_key in emo_audio_cache:
        cached_audio, cached_alpha = emo_audio_cache[cache_key]
        logging.info(f"  -> ✓ CACHE HIT: Reusing {Path(cached_audio).name} for {dimension}={value}")
        return cached_audio, cached_alpha

    logging.info(f"  -> Looking for emotion reference audio: {dimension}={value}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL HANDLING FOR GENDER MODE - CROSS-GENDER STYLE TRANSFER
    # ═══════════════════════════════════════════════════════════════════════════
    if dimension == "gender":
        logging.info(f"  ═══ GENDER MODE (CROSS-GENDER STYLE TRANSFER) ═══")
        logging.info(f"  -> Current speaker gender: {speaker_gender}, Target gender: {value}")

        if not speaker_gender:
            logging.warning(f"  -> ⚠️  Gender mode requires speaker gender, but none provided. Skipping.")
            return None, 1.0

        # Map current speaker gender + target gender → emotion reference directory
        # male speaker + (gender:female) → man_to_woman
        # female speaker + (gender:male) → woman_to_man
        if speaker_gender == "male" and value == "female":
            dir_name = "man_to_woman"
            logging.info(f"  -> Cross-gender transfer: male speaker pretending to be female → {dir_name}")
        elif speaker_gender == "female" and value == "male":
            dir_name = "woman_to_man"
            logging.info(f"  -> Cross-gender transfer: female speaker pretending to be male → {dir_name}")
        elif speaker_gender == value:
            # Same gender - no transfer needed
            logging.info(f"  -> Speaker is already {speaker_gender}, target is {value} - no transfer needed")
            return None, 1.0
        else:
            logging.warning(f"  -> Unknown gender combination: speaker={speaker_gender}, target={value}")
            return None, 1.0

        # Look for the emotion reference directory
        category_dir = Path(emotion_audio_pool_dir) / dir_name

        if not category_dir.exists():
            logging.warning(f"  -> ❌ Gender reference directory not found: {category_dir}")
            return None, 1.0

        # Get all audio files
        audio_files = list(category_dir.glob("*.mp3")) + list(category_dir.glob("*.wav"))

        if not audio_files:
            logging.warning(f"  -> ❌ No audio files found in {category_dir}")
            return None, 1.0

        # Randomly select one
        selected_audio = random.choice(audio_files)
        result_path = str(selected_audio)
        result_alpha = 1.0  # Use alpha=1.0 for strong gender style transfer

        # Save to cache
        if emo_audio_cache is not None:
            emo_audio_cache[cache_key] = (result_path, result_alpha)
            logging.info(f"  -> ✓ CACHE SAVE: {selected_audio.name} for {dimension}={value}")

        logging.info(f"  -> ✓ Selected gender reference: {selected_audio.name} from {category_dir}")
        return result_path, result_alpha

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK FOR "VERY" PREFIX - ONLY FOR EMOTION, SPEED, VOLUME, PITCH
    # ═══════════════════════════════════════════════════════════════════════════
    # IMPORTANT: "very" detection only applies to: emotion, speed, volume, pitch
    # NOT for gender or age tags
    very_dimensions = ["emotion", "speed", "volume", "pitch"]

    is_very = False
    if dimension in very_dimensions:
        # Check if value has "very_" or "very " prefix and determine alpha
        # Handle both "very_fast" (underscore) and "very afraid" (space) formats
        is_very = value.startswith("very_") or value.startswith("very ")

    emo_alpha = 1.0 if is_very else 0.6

    # Strip "very_" or "very " prefix for directory lookup
    if value.startswith("very_"):
        value_without_very = value.replace("very_", "", 1)
    elif value.startswith("very "):
        value_without_very = value.replace("very ", "", 1)
    else:
        value_without_very = value

    if is_very:
        logging.info(f"  -> Detected 'very' prefix on {dimension} - using alpha={emo_alpha} (full strength), searching for '{value_without_very}'")

    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL HANDLING FOR PITCH MODE - GENDER-AWARE REFERENCE SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    if dimension == "pitch":
        if not speaker_gender:
            logging.warning(f"  -> ⚠️  Pitch mode requires speaker gender, but none provided. Using default.")
            return None, emo_alpha

        # Map pitch values to gender-specific directory names
        # E.g., "high" + "male" → "high_pitch_man"
        #       "low" + "female" → "low_pitch_woman"
        pitch_value = value_without_very  # Use stripped value
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

        # Get directory name (use mapping if exists, otherwise use stripped value)
        dir_name = value_to_dir_map.get(value_without_very, value_without_very)

        if dir_name != value_without_very:
            logging.info(f"  -> Mapped '{value_without_very}' to directory '{dir_name}'")

    # Look for directory matching the tag value
    category_dir = Path(emotion_audio_pool_dir) / dir_name

    if not category_dir.exists():
        logging.warning(f"  -> ❌ Emotion reference directory not found: {category_dir} (will use default)")
        return None, emo_alpha

    # Get all audio files in the category (.mp3 or .wav)
    audio_files = list(category_dir.glob("*.mp3")) + list(category_dir.glob("*.wav"))

    if not audio_files:
        logging.warning(f"  -> ❌ No audio files found in {category_dir} (will use default)")
        return None, emo_alpha

    # Randomly select one audio file
    selected_audio = random.choice(audio_files)
    result_path = str(selected_audio)

    # Save to cache
    if emo_audio_cache is not None:
        emo_audio_cache[cache_key] = (result_path, emo_alpha)
        logging.info(f"  -> ✓ CACHE SAVE: {selected_audio.name} for {dimension}={value}")

    logging.info(f"  -> ✓ Selected emotion reference: {selected_audio.name} from {category_dir} (alpha={emo_alpha})")
    logging.info(f"     Full path: {selected_audio}")
    return result_path, emo_alpha

def _apply_accumulative_tts(
    text: str,
    para_tags: Dict[str, str],
    tts_model,
    initial_spk_audio: str,
    emotion_audio_pool_dir: str,
    speaker_gender: str = None,
    sample_rate: int = 24000,
    emo_audio_cache: Dict = None
) -> tuple:
    """
    Apply TTS with ACCUMULATIVE para tags pipeline.

    For each para tag:
    1. Generate audio with current_ref_speaker + para_audio_i
    2. Use generated audio as new ref_speaker for next iteration

    Pipeline:
        ref_speaker + para_audio_A → intermediate_A
        intermediate_A + para_audio_B → intermediate_B
        ...
        intermediate_(N-1) + para_audio_N → final_audio

    Args:
        text: Clean text to synthesize (no para tags)
        para_tags: Dictionary of dimension:value pairs (e.g., {"speed": "fast", "pitch": "high"})
        tts_model: Pre-initialized IndexTTS2 model
        initial_spk_audio: Path to initial speaker reference audio
        emotion_audio_pool_dir: Directory containing emotion reference audio pool
        speaker_gender: "male" or "female" (for pitch mode)
        sample_rate: Audio sample rate (default 24000)
        emo_audio_cache: Cache for emotion reference audio

    Returns:
        Tuple of (torch.Tensor, List[str]): Final audio and list of emotion reference paths used
    """
    if not para_tags:
        # No para tags - just do normal TTS
        temp_output = f"/tmp/accumulative_final.wav"
        tts_model.infer(
            spk_audio_prompt=initial_spk_audio,
            text=text,
            output_path=temp_output,
            use_random=False,
            verbose=False
        )
        wav, sr = torchaudio.load(temp_output)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return wav, []

    # Convert to list for ordered iteration
    para_tag_list = list(para_tags.items())
    num_tags = len(para_tag_list)

    logging.info(f"  ═══ ACCUMULATIVE TTS PIPELINE ═══")
    logging.info(f"  -> {num_tags} para tags to apply: {para_tags}")

    current_ref = initial_spk_audio
    temp_files = []
    emotion_references = []  # Track all emotion references used

    for idx, (dimension, value) in enumerate(para_tag_list):
        is_last = (idx == num_tags - 1)
        logging.info(f"  -> Step {idx + 1}/{num_tags}: Applying {dimension}={value}")

        # Get emotion reference audio for this dimension/value
        single_tag = {dimension: value}

        # Get emotion reference (now handles gender, pitch, and all other modes)
        # Uses cache: same tag reuses same audio file within dialogue
        emo_audio_prompt, emo_alpha = _get_emotion_reference_audio(
            single_tag, emotion_audio_pool_dir, speaker_gender=speaker_gender, emo_audio_cache=emo_audio_cache
        )

        if not emo_audio_prompt:
            logging.warning(f"  -> ❌ No emotion reference for {dimension}:{value}, skipping this tag")
            continue

        # Track this emotion reference
        emotion_references.append(str(emo_audio_prompt))

        # Generate audio with current reference + emotion reference
        temp_output = f"/tmp/accumulative_step_{idx}.wav"
        temp_files.append(temp_output)

        try:
            infer_kwargs = {
                "spk_audio_prompt": current_ref,
                "text": text,
                "output_path": temp_output,
                "use_random": False,
                "verbose": False
            }

            # Add emotion reference with alpha
            if emo_audio_prompt:
                infer_kwargs["emo_audio_prompt"] = emo_audio_prompt
                infer_kwargs["emo_alpha"] = emo_alpha
                logging.info(f"     Using emotion ref: {Path(emo_audio_prompt).name} (alpha={emo_alpha})")

            tts_model.infer(**infer_kwargs)

            # Load generated audio
            wav, sr = torchaudio.load(temp_output)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)

            # NOTE: Post-processing moved to the end (after all TTS steps)
            # This prevents TTS from washing away signal modifications like speed/volume

            if not is_last:
                # Not the last tag - save as intermediate and use as next reference
                intermediate_path = f"/tmp/accumulative_intermediate_{idx}.wav"
                torchaudio.save(intermediate_path, wav, sample_rate)
                current_ref = intermediate_path
                temp_files.append(intermediate_path)
                logging.info(f"     ✓ Step {idx + 1} complete → using as ref for next step")
            else:
                # Last tag - this is the final output
                logging.info(f"     ✓ Final step complete!")

        except Exception as e:
            logging.error(f"  -> ❌ Error at step {idx + 1}: {e}")
            continue

    # Clean up temp files (keep the final one until returned)
    for temp_file in temp_files[:-1]:  # Keep last one
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav[0:1, :]

    # Clean up the last temp file
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Apply post-processing (speed, volume) AFTER all TTS
    # ═══════════════════════════════════════════════════════════════════════════
    # Post-processing is applied ONCE at the end with ALL para_tags
    # This prevents TTS from washing away signal modifications
    logging.info(f"  ─── PHASE 2: POST-PROCESSING (all tags) ───")
    wav = _apply_audio_post_processing(wav, sample_rate, para_tags)

    logging.info(f"  ═══ ACCUMULATIVE PIPELINE COMPLETE ═══")
    logging.info(f"  -> Collected {len(emotion_references)} emotion references")
    return wav, emotion_references


# ═══════════════════════════════════════════════════════════════════════════
# TTS GENERATION HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _initialize_speakers(script, spk_audio_dir):
    """
    Initialize speaker audio files for User (A) and Agent (B).
    Handles gender mode pre-detection to set initial agent speaker opposite to first gender tag.

    Returns:
        Tuple of (spk_A_audio, spk_B_audio, spk_A_gender, spk_B_gender)
    """
    # Pre-detect first gender tag
    first_gender_tag = _pre_detect_first_gender_tag(script)

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
        # Gender mode: set initial agent to opposite gender
        initial_agent_gender = _get_opposite_gender(first_gender_tag)
        if initial_agent_gender:
            logging.info(f"  ═══ GENDER MODE INITIALIZATION ═══")
            logging.info(f"  -> First gender switch will be to: {first_gender_tag}")
            logging.info(f"  -> Setting initial agent gender to: {initial_agent_gender} (opposite)")

            gender_dir = Path(spk_audio_dir) / initial_agent_gender
            gender_audio_files = list(gender_dir.glob("*.wav"))
            if gender_audio_files:
                spk_B_audio = str(random.choice(gender_audio_files))
                spk_B_gender = initial_agent_gender
                logging.info(f"  -> ✓ Initial agent speaker set: {Path(spk_B_audio).name} ({initial_agent_gender})")
            else:
                logging.warning(f"  -> ❌ No audio files in {gender_dir}, using random")
                spk_B_audio = str(random.choice(remaining))
                spk_B_gender = _detect_speaker_gender(spk_B_audio, spk_audio_dir)
        else:
            spk_B_audio = str(random.choice(remaining))
            spk_B_gender = _detect_speaker_gender(spk_B_audio, spk_audio_dir)
    else:
        # No gender mode: random selection
        spk_B_audio = str(random.choice(remaining))
        spk_B_gender = _detect_speaker_gender(spk_B_audio, spk_audio_dir)

    logging.info(f"Speaker A (User): {spk_A_audio}")
    logging.info(f"Speaker B (Agent): {spk_B_audio}")

    # Detect gender for User speaker
    spk_A_gender = _detect_speaker_gender(spk_A_audio, spk_audio_dir)

    return spk_A_audio, spk_B_audio, spk_A_gender, spk_B_gender


def _normalize_para_tags(para_tags, multi_para_mode):
    """
    Normalize paralinguistic tags and determine if accumulative pipeline should be used.

    Returns:
        Tuple of (normalized_para_tags, use_accumulative)
    """
    if not para_tags:
        return para_tags, False

    num_tags = len(para_tags)

    # Multi-para mode with multiple tags - use accumulative pipeline
    if multi_para_mode and num_tags > 1:
        logging.info(f"  ═══ MULTI-PARA ACCUMULATIVE MODE ({num_tags} tags) ═══")
        logging.info(f"  -> Tags: {para_tags}")
        return para_tags, True

    # Single tag or multi-para disabled
    return para_tags, False


def _generate_tts_audio(tts_model, clean_text, para_tags, spk_audio, spk_gender,
                        emo_audio_prompt, emo_alpha, use_accumulative, emotion_audio_pool_dir, sample_rate,
                        emo_audio_cache=None):
    """
    Generate TTS audio for a single turn.
    Handles both accumulative (multi-para) and single-tag pipelines.

    Args:
        emo_alpha: Emotion reference strength (1.0 for "very_*" tags, 0.6 for normal)
        emo_audio_cache: Cache for emotion reference audio (same tag reuses same audio)

    Returns:
        Tuple of (Audio tensor, List of emotion reference paths)
        - For single-tag mode: returns (wav, [single_emotion_ref] or [])
        - For accumulative mode: returns (wav, [emotion_ref1, emotion_ref2, ...])
    """
    temp_output = f"/tmp/indextts_turn_{random.randint(0, 999999)}.wav"

    try:
        # Branch based on pipeline type
        if use_accumulative and para_tags and len(para_tags) > 1:
            # Multi-para accumulative pipeline
            logging.info(f"  -> Using ACCUMULATIVE pipeline for {len(para_tags)} tags")
            wav, emotion_references = _apply_accumulative_tts(
                text=clean_text,
                para_tags=para_tags,
                tts_model=tts_model,
                initial_spk_audio=spk_audio,
                emotion_audio_pool_dir=emotion_audio_pool_dir,
                speaker_gender=spk_gender,
                sample_rate=sample_rate,
                emo_audio_cache=emo_audio_cache
            )
        else:
            # Single-tag TTS
            wav = _infer_single_tag_tts(
                tts_model, clean_text, spk_audio, emo_audio_prompt, emo_alpha,
                temp_output, para_tags, sample_rate
            )
            # For single-tag mode, return single emotion reference as list
            emotion_references = [str(emo_audio_prompt)] if emo_audio_prompt else []

        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav[0:1, :]

        return wav, emotion_references

    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        # Return silence as fallback
        return torch.zeros(1, sample_rate), []
    finally:
        # Cleanup temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)


def _infer_single_tag_tts(tts_model, clean_text, spk_audio, emo_audio_prompt, emo_alpha,
                          temp_output, para_tags, sample_rate):
    """
    Run single-tag TTS inference with optional emotion reference and post-processing.

    Args:
        emo_alpha: Emotion reference strength (1.0 for "very_*" tags, 0.6 for normal)

    Returns:
        Audio tensor (channels, samples)
    """
    # Build inference kwargs
    infer_kwargs = {
        "spk_audio_prompt": spk_audio,
        "text": clean_text,
        "output_path": temp_output,
        "use_random": False,
        "verbose": False
    }

    # Add emotion reference if available
    if emo_audio_prompt is not None:
        infer_kwargs["emo_audio_prompt"] = emo_audio_prompt
        infer_kwargs["emo_alpha"] = emo_alpha
        logging.info(f"  -> 🎵 Using emotion reference: {Path(emo_audio_prompt).name} (alpha={emo_alpha})")

    # Run inference
    tts_model.infer(**infer_kwargs)

    # Load generated audio
    wav, sr = torchaudio.load(temp_output)

    # Resample if necessary
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav[0:1, :]

    # Apply post-processing (speed, volume)
    if para_tags:
        wav = _apply_audio_post_processing(wav, sample_rate, para_tags)
        logging.info(f"  -> ✓ Post-processing applied")

    return wav


def _combine_audio_segments(audio_segments, sample_rate):
    """
    Combine audio segments into stereo (User=left, Agent=right).
    Handles normal concatenation and overlap.

    Returns:
        Stereo audio tensor (2, samples)
    """
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
                # Overlap - reduce previous audio
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
    return torch.cat([l_ch, r_ch], dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TTS GENERATION FUNCTION - Refactored for clarity
# ═══════════════════════════════════════════════════════════════════════════

def IndexTTS_gen(script, output, tts_model, spk_audio_dir="examples", emotion_audio_pool_dir=None, multi_para_mode=False, save_individual_turns=False, turn_output_dir=None):
    """
    Generate TTS audio using IndexTTS2 model with emotion reference audio.

    REFACTORED VERSION - Much cleaner and more readable!

    Architecture:
        - Timbre Prompt (spk_audio_prompt): Controls speaker voice identity (stays constant)
        - Style Prompt (emo_audio_prompt): Controls tone/style from reference audio

    Modes:
        - Multi-para: Multiple tags with accumulative pipeline
        - Gender: Cross-gender style transfer via emotion reference (man_to_woman / woman_to_man)
                  Speaker voice stays the same, only style changes
        - Pitch: Gender-aware emotion reference
        - Other: Emotion reference (emotion, speed, volume, age)

    Args:
        script: List of (role, text) tuples where text may contain (tags)
        output: Path to save the output wav file
        tts_model: Pre-initialized IndexTTS2 model instance
        spk_audio_dir: Directory containing speaker reference audio files
        emotion_audio_pool_dir: Directory containing emotion reference audio pool
        multi_para_mode: If True, use accumulative pipeline for multiple para tags
        save_individual_turns: If True, save each turn as a separate mono audio file
        turn_output_dir: Directory to save individual turn audio files (required if save_individual_turns=True)

    Returns:
        If save_individual_turns=True, returns a list of metadata dicts for each turn:
        [{"turn_idx": int, "role": str, "text": str, "para_tags": dict, "audio_path": str,
          "audio_start": float, "audio_end": float, "audio_duration": float,
          "speaker_reference": str, "emotion_reference": list[str]}]

        Note: emotion_reference is now a list of emotion reference audio paths:
        - For single-tag mode: list with one path or empty list
        - For multi-para mode: list with multiple paths (one per tag)

        Otherwise returns None
    """
    sample_rate = 24000

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Initialize speakers (handles gender mode pre-detection)
    # ═══════════════════════════════════════════════════════════════════════════
    spk_A_audio, spk_B_audio, spk_A_gender, spk_B_gender = _initialize_speakers(script, spk_audio_dir)

    # Track current speakers (fixed throughout dialogue - gender changes via emotion reference)
    current_speakers = {
        "A_audio": spk_A_audio,
        "B_audio": spk_B_audio,
        "A_gender": spk_A_gender,
        "B_gender": spk_B_gender
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # EMOTION AUDIO CACHE - Same tag reuses same audio within dialogue
    # Key: (dimension, value), Value: (audio_path, emo_alpha)
    # ═══════════════════════════════════════════════════════════════════════════
    emo_audio_cache = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Process each dialogue turn
    # ═══════════════════════════════════════════════════════════════════════════
    audio_segments = []
    turn_metadata_list = []  # For tracking metadata when save_individual_turns=True
    current_time = 0.0  # Track cumulative time for metadata

    # Create turn output directory if needed
    if save_individual_turns:
        if turn_output_dir is None:
            raise ValueError("turn_output_dir must be specified when save_individual_turns=True")
        turn_output_path = Path(turn_output_dir)
        turn_output_path.mkdir(parents=True, exist_ok=True)

    for idx, (role, text) in enumerate(script, 1):
        # 2.1 Determine speaker for this role
        is_user = role in ("User", "[overlap] User")
        spk_audio = current_speakers["A_audio"] if is_user else current_speakers["B_audio"]
        spk_gender = current_speakers["A_gender"] if is_user else current_speakers["B_gender"]

        # 2.2 Extract clean text and paralinguistic tags
        clean_text, para_tags = extract_paralinguistic_tags(text)
        if para_tags:
            logging.info(f"Turn {idx} ({role}): Extracted tags {para_tags}")

        # 2.3 Normalize tags (handle multi-para edge cases)
        para_tags, use_accumulative = _normalize_para_tags(para_tags, multi_para_mode)

        # 2.4 Get emotion reference audio (handles gender via cross-gender style transfer)
        # Note: Speaker voice stays the same, gender is applied via emotion reference
        # Uses cache: same tag reuses same audio file within dialogue
        emo_audio_prompt, emo_alpha = _get_emotion_reference_audio(
            para_tags, emotion_audio_pool_dir, speaker_gender=spk_gender, emo_audio_cache=emo_audio_cache
        )

        # 2.5 Generate audio for this turn
        wav, emotion_references = _generate_tts_audio(
            tts_model, clean_text, para_tags, spk_audio, spk_gender,
            emo_audio_prompt, emo_alpha, use_accumulative, emotion_audio_pool_dir, sample_rate,
            emo_audio_cache=emo_audio_cache
        )

        # 2.6 Save individual turn audio if requested
        if save_individual_turns:
            # Calculate turn duration
            turn_duration = wav.shape[1] / sample_rate
            audio_start = current_time
            audio_end = current_time + turn_duration
            current_time = audio_end

            # Save mono turn audio
            # Simple filename: turn00.wav, turn01.wav, etc.
            turn_audio_path = turn_output_path / f"turn{idx-1:02d}.wav"

            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav_mono = wav.mean(dim=0, keepdim=True)
            else:
                wav_mono = wav

            torchaudio.save(str(turn_audio_path), wav_mono, sample_rate)

            # Store metadata
            # Convert para_tags to structured format (all 6 dimensions)
            para_tags_structured = {
                "gender": None,
                "age": None,
                "pitch": None,
                "speed": None,
                "volume": None,
                "emotion": None
            }
            if para_tags:
                for dim, val in para_tags.items():
                    para_tags_structured[dim] = val

            # Store emotion references as list
            # For multi-para mode: multiple references
            # For single-tag mode: single reference as list (or empty list)
            emotion_refs_list = emotion_references if emotion_references else []

            turn_metadata = {
                "turn_idx": idx - 1,  # 0-indexed
                "role": role,
                "speaker": "user" if is_user else "agent",
                "text": clean_text,
                "para_tags": para_tags_structured,
                "audio_path": str(turn_audio_path),
                "audio_start": audio_start,
                "audio_end": audio_end,
                "audio_duration": turn_duration,
                "speaker_reference": str(spk_audio),
                "emotion_reference": emotion_refs_list
            }
            turn_metadata_list.append(turn_metadata)

        # 2.7 Store audio segment
        audio_segments.append((role, wav, spk_A_audio))

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: Combine into stereo (User=left, Agent=right)
    # ═══════════════════════════════════════════════════════════════════════════
    full_dialog = _combine_audio_segments(audio_segments, sample_rate)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: Save audio file
    # ═══════════════════════════════════════════════════════════════════════════
    torchaudio.save(str(output), full_dialog, sample_rate)
    logging.info(f"Saved TTS audio to: {output}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: Return metadata if individual turns were saved
    # ═══════════════════════════════════════════════════════════════════════════
    if save_individual_turns:
        # Also save the full dialogue path in metadata
        for metadata in turn_metadata_list:
            metadata["full_dialogue_audio"] = str(output)
        return turn_metadata_list
    return None

def _init_tts_model(cfg):
    """Initialize IndexTTS2 model once; pass the returned model into tts_batch."""
    model_dir = cfg.tts.get("model_dir", "checkpoints")
    cfg_path = os.path.join(model_dir, "config.yaml")

    logging.info("Initializing IndexTTS2 model with optimizations...")
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
        device="cuda",
        use_fp16=True,
        use_cuda_kernel=True,
        use_accel=False,
        use_torch_compile=False,
        use_deepspeed=False
    )
    logging.info("IndexTTS2 model initialized successfully with all optimizations!")
    return tts_model


def tts_batch(cfg, topic, mode, tts_model):
    """
    Batch process dialogue text files to generate TTS audio using IndexTTS2.
    Processes files from dialogue output directory and extracts paralinguistic tags.
    Uses dual-node approach:
    - Emotion tags → Emotion vectors
    - Gender/Age tags → Emotion reference audio
    """
    src_dir = Path(cfg.data_root) / mode / topic / "dialogue_txt"
    wav_dir = Path(cfg.data_root) / mode / topic / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Speaker audio directory and emotion audio pool from config
    spk_audio_dir = cfg.tts.get("spk_audio_dir", "examples")
    emotion_audio_pool_dir = cfg.tts.get("emotion_audio_pool_dir", None)

    # Multi-para mode derived from the mode parameter
    multi_para_mode = (mode == "multi")
    if multi_para_mode:
        logging.info("=" * 60)
        logging.info("MULTI-PARA MODE ENABLED - Using accumulative TTS pipeline")
        logging.info("=" * 60)

    # Create error log file
    error_log_path = wav_dir / "tts_errors.txt"

    # Check if HuggingFace export is enabled to decide whether to save individual turns
    save_individual_turns = cfg.get("huggingface", {}).get("enabled", False)
    if save_individual_turns:
        logging.info("HuggingFace export enabled - saving individual turn audio files")

    # Get all dialogue text files
    dialogue_files = list(src_dir.glob("*.txt"))

    if not dialogue_files:
        logging.warning(f"No dialogue files found in {src_dir}")
        return

    logging.info(f"Processing {len(dialogue_files)} dialogue files for TTS [{mode}/{topic}]...")

    for txt_file in tqdm(dialogue_files, desc="TTS Generation"):
        dialogue_id = txt_file.stem

        # Create dialogue folder: wav_dir/dialogue_id/
        dialogue_folder = wav_dir / dialogue_id
        dialogue_folder.mkdir(parents=True, exist_ok=True)

        # Output paths
        wav_file_path = dialogue_folder / "full.wav"

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

            # Prepare turn output directory if saving individual turns
            # Structure: wav_dir/dialogue_id/individual/
            if save_individual_turns:
                dialogue_turn_dir = dialogue_folder / "individual"
                dialogue_turn_dir.mkdir(parents=True, exist_ok=True)
            else:
                dialogue_turn_dir = None

            turn_metadata = IndexTTS_gen(
                script,
                wav_file_path,
                tts_model=tts_model,  # Pass pre-initialized model
                spk_audio_dir=spk_audio_dir,
                emotion_audio_pool_dir=emotion_audio_pool_dir,  # Pass emotion audio pool
                multi_para_mode=multi_para_mode,  # Pass multi-para mode flag from config
                save_individual_turns=save_individual_turns,
                turn_output_dir=dialogue_turn_dir
            )

            # Save metadata per dialogue if available
            if save_individual_turns and turn_metadata:
                metadata_file = dialogue_turn_dir / "turn_metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(turn_metadata, f, indent=2, ensure_ascii=False)
                logging.info(f"Saved turn metadata to: {metadata_file}")

        except Exception as e:
            # Log error and continue with next file
            error_msg = f"Error processing {txt_file.name}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            with open(error_log_path, "a", encoding="utf-8") as error_file:
                error_file.write(f"{txt_file.name}: {str(e)}\n")
            continue

    logging.info(f"TTS batch processing complete. Audio files saved to: {wav_dir}")

# ─────────────────────────────  HUGGINGFACE DATASET EXPORT  ──────────────────────────────

def export_to_huggingface(cfg):
    """
    Export generated dialogues and audio to HuggingFace dataset format.
    Each row represents a single dialogue turn with full metadata.

    Audio organization:
    - Each dialogue saved in its own folder
    - Each turn as mono audio file
    - Full dialogue as stereo audio file

    This function reads the metadata generated during TTS processing.
    """
    try:
        from datasets import Dataset, Features, Value, Sequence
    except ImportError:
        logging.error("HuggingFace datasets library not installed. Install with: pip install datasets")
        return

    # Check if HuggingFace export is enabled
    if not cfg.get("huggingface", {}).get("enabled", False):
        logging.info("HuggingFace dataset export disabled in config")
        return

    logging.info("=" * 80)
    logging.info("EXPORTING TO HUGGINGFACE DATASET FORMAT")
    logging.info("=" * 80)

    # Get configuration
    data_root = Path(cfg.data_root)
    output_dir = data_root / "huggingface_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all (mode, topic) wav directories from config
    topics = list(cfg.get("topics", []))
    modes = list(cfg.get("dialogue_modes", []))

    # Pre-load scenarios per topic
    topic_scenarios = {}
    for topic in topics:
        scen_file = data_root / "scenarios" / topic / "scenarios.jsonl"
        if scen_file.exists():
            topic_scenarios[topic] = {}
            for line in scen_file.read_text(encoding="utf-8").splitlines():
                s = json.loads(line)
                topic_scenarios[topic][s["id"]] = s

    # Collect all (mode, topic, dialogue_folder, scenarios, wav_dir) tuples
    all_dialogue_items = []
    for mode in modes:
        for topic in topics:
            wav_dir = data_root / mode / topic / "wav"
            if not wav_dir.exists():
                continue
            scenarios = topic_scenarios.get(topic, {})
            for d in sorted(wav_dir.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    all_dialogue_items.append((mode, topic, d, scenarios, wav_dir))

    if not all_dialogue_items:
        logging.error(f"No dialogue folders found under {data_root}")
        logging.error("Please run the TTS stage first with HuggingFace export enabled")
        return

    logging.info(f"Found {len(all_dialogue_items)} dialogue folders across all modes/topics")

    # Get LLM names
    llm_user = cfg.dialogue.get("user_model", "unknown")
    llm_agent = cfg.dialogue.get("agent_model", "unknown")

    # Collect all dialogue data from metadata
    dataset_rows = []

    for mode, topic, dialogue_folder, scenarios, wav_dir in tqdm(all_dialogue_items, desc="Building HuggingFace dataset"):
        dialogue_id = dialogue_folder.name

        # Check for metadata file
        metadata_file = dialogue_folder / "individual" / "turn_metadata.json"
        if not metadata_file.exists():
            logging.warning(f"No metadata file found for {dialogue_id}, skipping")
            continue

        # Load metadata for this dialogue
        with open(metadata_file, "r", encoding="utf-8") as f:
            turn_list = json.load(f)

        # Parse dialogue_id: {topic}_{mode}_{scenario_idx}_{dialogue_idx}
        # topic and mode already known from outer loop; just parse scenario/conversation ids
        parts = dialogue_id.split("_")
        if len(parts) >= 4:
            scenario_idx = parts[2]
            scenario_id = f"scenario{scenario_idx}"
            scenario_description = scenarios.get(scenario_id, {}).get("description", None)
            conversation_id = f"{topic}_{mode}_{scenario_idx}"
        else:
            scenario_description = None
            conversation_id = dialogue_id

        # Full dialogue audio path
        full_dialogue_path = dialogue_folder / "full.wav"

        # Process each turn using the metadata
        for turn_metadata in turn_list:
            # Flatten para_tags into individual fields
            para_tags = turn_metadata["para_tags"]

            # Handle emotion_reference - ensure it's always a list
            emotion_ref_raw = turn_metadata.get("emotion_reference", None)
            if emotion_ref_raw is None:
                emotion_ref_list = []
            elif isinstance(emotion_ref_raw, list):
                # Already a list (new format)
                emotion_ref_list = emotion_ref_raw
            elif isinstance(emotion_ref_raw, str):
                # Old format - single string, convert to list
                emotion_ref_list = [emotion_ref_raw] if emotion_ref_raw else []
            else:
                emotion_ref_list = []

            # Create dataset row from metadata
            row = {
                "conversation_id": conversation_id,
                "mode": mode,
                "topic": topic,
                "scenario": scenario_description,
                "turn_index": turn_metadata["turn_idx"],
                "audio_path": str(Path(turn_metadata["audio_path"]).relative_to(data_root)),
                "LLM1": llm_user,
                "LLM2": llm_agent,
                "speaker": turn_metadata["speaker"],
                "text": turn_metadata["text"],
                "paralinguistic_info": {
                    "gender": para_tags.get("gender", None),
                    "age": para_tags.get("age", None),
                    "pitch": para_tags.get("pitch", None),
                    "speed": para_tags.get("speed", None),
                    "volume": para_tags.get("volume", None),
                    "emotion": para_tags.get("emotion", None),
                },
                "audio": turn_metadata["audio_path"],
                "reference": turn_metadata.get("speaker_reference", None),
                "emotion_reference": emotion_ref_list,
                "audio_start": turn_metadata["audio_start"],
                "audio_end": turn_metadata["audio_end"],
                "audio_duration": turn_metadata["audio_duration"],
                "full_dialogue_audio": str(full_dialogue_path)
            }

            dataset_rows.append(row)

    # Create HuggingFace dataset
    logging.info(f"Creating HuggingFace dataset with {len(dataset_rows)} rows...")

    # Define features schema
    # Note: We store audio as string paths initially to avoid torchcodec dependency
    # Users can cast to Audio later with: dataset = dataset.cast_column("audio", Audio())
    features = Features({
        "conversation_id": Value("string"),
        "mode": Value("string"),
        "turn_index": Value("int32"),
        "audio_path": Value("string"),
        "topic": Value("string"),
        "scenario": Value("string"),
        "LLM1": Value("string"),
        "LLM2": Value("string"),
        "speaker": Value("string"),
        "text": Value("string"),
        "paralinguistic_info": {
            "gender": Value("string"),
            "age": Value("string"),
            "pitch": Value("string"),
            "speed": Value("string"),
            "volume": Value("string"),
            "emotion": Value("string"),
        },
        "audio": Value("string"),  # Store as string path, cast to Audio() later if needed
        "reference": Value("string"),
        "emotion_reference": Sequence(Value("string")),  # List of emotion reference paths (for multi-para mode)
        "audio_start": Value("float32"),
        "audio_end": Value("float32"),
        "audio_duration": Value("float32"),
        "full_dialogue_audio": Value("string")
    })

    # Create dataset
    dataset = Dataset.from_list(dataset_rows, features=features)

    # Optionally cast audio column to Audio feature if torchcodec is available
    try:
        from datasets import Audio as AudioFeature
        dataset = dataset.cast_column("audio", AudioFeature(sampling_rate=24000))
        logging.info("Successfully cast audio column to Audio feature type")
    except ImportError:
        logging.info("Audio stored as file paths. To load audio, install torchcodec and use: dataset.cast_column('audio', Audio())")
    except Exception as e:
        logging.warning(f"Could not cast audio column to Audio feature: {e}")
        logging.info("Audio stored as file paths. You can manually cast later if needed.")

    # Save dataset
    dataset_save_path = output_dir / "dataset"
    dataset.save_to_disk(str(dataset_save_path))
    logging.info(f"Dataset saved to: {dataset_save_path}")

    # Also save as JSON for easy inspection
    json_path = output_dir / "dataset_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Remove audio binary data for JSON export
        json_rows = []
        for row in dataset_rows:
            row_copy = row.copy()
            row_copy["audio"] = str(row_copy["audio"])  # Convert to string path
            json_rows.append(row_copy)
        json.dump(json_rows, f, indent=2, ensure_ascii=False)
    logging.info(f"Metadata saved to: {json_path}")

    # Push to HuggingFace Hub if requested
    if cfg.huggingface.get("push_to_hub", False):
        hub_repo_id = cfg.huggingface.get("hub_repo_id")
        if hub_repo_id:
            logging.info(f"Pushing dataset to HuggingFace Hub: {hub_repo_id}")
            dataset.push_to_hub(
                hub_repo_id,
                private=cfg.huggingface.get("hub_private", False)
            )
            logging.info("Dataset successfully pushed to HuggingFace Hub!")
        else:
            logging.warning("push_to_hub is True but hub_repo_id not specified")

    logging.info("=" * 80)
    logging.info("HUGGINGFACE DATASET EXPORT COMPLETE")
    logging.info(f"Total turns exported: {len(dataset_rows)}")
    logging.info(f"Dataset location: {dataset_save_path}")
    logging.info("=" * 80)

# ─────────────────────────────  ORCHESTRATOR  ──────────────────────────────

@dataclass
class PipelineConfig:
    # device
    device: str = "cuda"

    # root directory; all outputs go to data_root / mode / topic / ...
    data_root: str = "data_para"

    # topics to generate in one run
    topics: List[str] = field(default_factory=list)

    # dialogue modes: dimension names + "multi"
    dialogue_modes: List[str] = field(default_factory=lambda: ["multi"])

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

    # huggingface dataset export
    huggingface: OmegaConf = OmegaConf.create({
        "enabled": True,
        "dataset_name": "synthetic_paralinguistic_dialogues",
        "output_dir": "data_para/huggingface_dataset",
        "push_to_hub": False,
        "hub_repo_id": None,
        "hub_private": False,
    })

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        print("DEBUG stages =", self.cfg.stages)

    def run(self):
        st = set(self.cfg.stages)
        data_root = Path(self.cfg.data_root)
        topics = list(self.cfg.get("topics", []))
        modes = list(self.cfg.get("dialogue_modes", []))

        if "scenario" in st:
            for topic in topics:
                print(f"Generating scenarios: {topic}")
                generate_scenarios(self.cfg, topic)
                out_path = data_root / "scenarios" / topic / "scenarios.json"
                convert_nested_json_to_jsonl(out_path, out_path.with_suffix(".jsonl"))

        if "dialogue" in st:
            for topic in topics:
                for mode in modes:
                    print(f"Generating dialogues: topic={topic}, mode={mode}")
                    generate_paralinguistic_dialogues(self.cfg, topic, mode)

        if "tts" in st:
            print("Initializing TTS model (once for all topics/modes)...")
            tts_model = _init_tts_model(self.cfg)
            for topic in topics:
                for mode in modes:
                    print(f"TTS: topic={topic}, mode={mode}")
                    tts_batch(self.cfg, topic, mode, tts_model)

        if "huggingface" in st:
            print("Exporting to HuggingFace Dataset...")
            export_to_huggingface(self.cfg)

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

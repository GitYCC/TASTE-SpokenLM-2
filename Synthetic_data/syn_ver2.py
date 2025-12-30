from __future__ import annotations

import json
import re
import os
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
# import torch  # Commented out - only needed for TTS
from omegaconf import OmegaConf
from tqdm import tqdm
from openai import OpenAI
import hydra, sys
import json
from pathlib import Path
# import torchaudio  # Commented out - only needed for TTS
import time
import json
import httpx._content
from openai import AzureOpenAI
from dotenv import load_dotenv
import httpx


# ═══════════════════════════════════════════════════════════════════════════
# TTS IMPORTS (COMMENTED OUT - Uncomment when you need TTS functionality)
# ═══════════════════════════════════════════════════════════════════════════

# # TTS Env - XTTS
# # from TTS.tts.configs.xtts_config import XttsConfig
# # torch.serialization.add_safe_globals([XttsConfig])
# # from TTS.api import TTS

# # CosyVoice Env
# os.environ["COSYVOICE_NO_AUTO_DOWNLOAD"] = "1"  # 若使用 CosyVoice ≥ 0.5.2
# from modelscope import snapshot_download
# # snapshot_download(
# #     'iic/CosyVoice-300M',
# #     local_dir='pretrained_models/CosyVoice-300M',
# #     revision='master'
# # )
# snapshot_download(
#     'iic/CosyVoice2-0.5B',
#     local_dir='pretrained_models/CosyVoice2-0.5B',
#     revision='master'
# )
# import sys, pathlib
# MATCHA = pathlib.Path(__file__).resolve().parent / "third_party" / "Matcha-TTS"
# src = MATCHA / "src"
# sys.path.insert(0, str(src if src.exists() else MATCHA))
# ROOT = pathlib.Path(__file__).resolve().parent
# CV_DIR = ROOT / "CosyVoice"
# if not (CV_DIR / "cosyvoice").exists() and (CV_DIR / "src" / "cosyvoice").exists():
#     CV_DIR = CV_DIR / "src"
# sys.path.insert(0, str(CV_DIR))
# from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
# from cosyvoice.utils.file_utils import load_wav
# import unicodedata
# sys.path.append('CosyVoice/third_party/Matcha-TTS')
# os.environ['PYTHONPATH'] = 'CosyVoice/third_party/Matcha-TTS:' + os.environ.get('PYTHONPATH', '')
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import unicodedata  # Moved here - used by ascii_only function

APIKEY = os.getenv("OPENROUTER_API_KEY")


# ──────────────────────────────  Normal UTILS  ────────────────────────────────
def ascii_only(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def convert_nested_json_to_jsonl(input_path, output_path):
    # 1. Read the entire JSON object
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError("No 'scenarios' key or empty list in the input JSON.")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in scenarios:
            # Each entry looks like {"scenario1": {...}}; get the key and the value
            scenario_id, inner = next(iter(entry.items()))
            # Write a flat dict
            out_obj = {
                "id": scenario_id,
                "description": inner["description"]
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Converted {len(scenarios)} scenarios to JSONL → {output_path}")

def format_headers_in_lines(text):
    # Add a newline before [header] if it's not already at the start of a line
    return re.sub(r'(\\[[^\\]]+\\])\s*\n\s*', r'\1 ',text)


def merge_overlapping_user_lines(lines):
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is a User line and the next line is an [overlap] User line
        if line.startswith("User:") and i + 1 < len(lines) and lines[i + 1].startswith("[overlap] User:"):
            # Remove the label from the overlap line and merge
            merged_text = line.rstrip() + " " + lines[i + 1].replace("[overlap] User:", "", 1).lstrip()
            merged_lines.append("User:" + merged_text[len("User:"):])
            i += 2  # Skip the next line as it's merged
        else:
            merged_lines.append(line)
            i += 1
    return merged_lines

import re
from pathlib import Path

def split_and_save_dialogues(llm_output, out_dir, base_filename="scenarios"):
    """
    Splits the LLM output by 'Dialogue 1:', 'Dialogue2:', etc.,
    removes the header, and saves each as a separate .txt file in out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Updated pattern to match with or without space
    pattern = r'(Dialogue\s*\d+:)'
    matches = list(re.finditer(pattern, llm_output))
    splits = [m.start() for m in matches]
    splits.append(len(llm_output))  # Add end of string

    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i+1]
        dialogue_text = llm_output[start:end].strip()
        # Remove the first line (the Dialogue header)
        lines = dialogue_text.splitlines()
        if lines:
            lines = lines[1:]  # Remove the header
        dialogue_body = "\n".join(lines).strip()
        filename = out_dir / f"{base_filename}_{i+1}.txt"
        filename.write_text(dialogue_body, encoding="utf-8")
        print(f"Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════
# TTS FUNCTIONS (COMMENTED OUT - Uncomment when you need TTS functionality)
# ═══════════════════════════════════════════════════════════════════════════

# def CosyVoice_gen(mode,script,output):
# #     if mode =="emotion" or mode=="human":
# #         cosyvoice = CosyVoice2(
# #         "pretrained_models/CosyVoice2-0.5B",
# #         load_jit=False, load_trt=False, fp16=False
# #         )
# #         spk_A = 'Rosemary Okafor'
# #         random.seed(None)
# #         random.seed(time.time_ns())
# 
# #         all_speakers = ['Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Xavier Hayasaka', 'Luis Moray','Alison Dietlinde', 'Alexandra Hisakawa', 'Ana Florence', 'Asya Anara',  'Andrew Chipper', 'Annmarie Nele', 'Badr Odhiambo', 'Barbora MacLean', 'Brenda Stern',  'Chandra MacFarland', 'Claribel Dervla', 'Daisy Studious', 'Gitta Nikolina', 'Gracie Wise', 'Henriette Usha', 'Lidiya Szekeres', 'Lilya Stainthorpe', 'Maja Ruoho', 'Nova Hogarth', 'Narelle Moon', 'Rosemary Okafor', 'Sofia Hellen', 'Szofi Granger', 'Suad Qasim', 'Tammie Ema', 'Tammy Grit', 'Tanja Adelina', 'Uta Obando', 'Vjollca Johnnie'] #Ferran_simen, Ludvig Milivoj, Marcos Rudask , 'Zofija Kendrick' , 'Alma María', 'Camilla Holmström', 'Eugenio Mataracı'
# 
# #         spk_B = random.choice(all_speakers)
# #     else:
# #         cosyvoice = CosyVoice(
# #         "pretrained_models/CosyVoice-300M_SFT",
# #         load_jit=False, load_trt=False, fp16=False
# #         )
# #         all_speakers = ["英文女" ,"英文男"]
# #         spk_A = random.choice(all_speakers)
# #         spk_B = "英文女" if spk_A == "英文男" else "英文男"
# 
# 
# #     for idx, (role, text) in enumerate(script, 1):
# #         spk_id = spk_B if role in ("User", "[overlap] User","[pause] User","[backchannel] User") else spk_A
# #         ref_id = spk_id.replace(" ", "_")
# #         ref_id = ascii_only(ref_id)
# #         emotion = ""
# #         print(spk_id)
# #         if text.startswith("("):
# #                 closing = text.find(")")
# #                 if closing != -1:
# #                     emotion = text[1:closing].strip()
# #                     text = text[closing+1:].strip()
# #         if text != "[pause]":
# #             text = text.replace("[pause]", "")
# 
# 
# #         prompt_speech =  load_wav(f"XTTS_wav/all/sample_{ref_id}.wav", 16000)
# #         if mode =="emotion":
# #             output = str(output)
# #             chunks = list(
# #                 cosyvoice.inference_instruct2(text, emotion, prompt_speech, stream=False)
# #             )
# #         elif mode == "human":
# #             output = str(output)
# #             chunks = list(
# #                 cosyvoice.inference_cross_lingual(text, prompt_speech, stream=False)
# #             )
# #         else:
# #             chunks = list(
# #                 cosyvoice.inference_sft(text, spk_id, stream=False)
# #             )
# #         wav = torch.cat([c["tts_speech"] for c in chunks], -1)
# #         if wav.ndim == 1:
# #             wav = wav.unsqueeze(0)
# 
# #         pause = torch.zeros(1, int(random.uniform(0.25, 0.4) * cosyvoice.sample_rate))
# #         if role.strip().lower().startswith("[pause]"):
# #             wav = torch.zeros(1,1)
# #             pause = torch.zeros(1, int(random.uniform(0.6, 1.0) * cosyvoice.sample_rate))
# #         elif role.strip().lower().startswith("[backchannel]"):
# #             pause = torch.zeros(1, int(random.uniform(0.05, 0.15) * cosyvoice.sample_rate))
# 
# #         if idx == 1:
# #             # first utterance
# #             r_ch = wav
# #             l_ch = torch.zeros_like(wav)
# #         else:
# #             if role.strip().lower().startswith("[overlap]"):
# #                 overlap_frame = int(random.uniform(1.0, 1.6) * cosyvoice.sample_rate)
# #                 padded = torch.zeros_like(wav)
# #                 if spk_id == spk_A:
# #                     # Slice l_ch for overlap and pad r_ch
# #                     l_ch = l_ch[:, : -overlap_frame]  # Ensure l_ch has overlap frame removed
# #                     l_ch = torch.cat([l_ch, pause, wav], -1)  # Concatenate pause + new speech for l_ch
# #                     r_ch = torch.cat([r_ch, pause, padded], -1)  # Add silence to r_ch
# #                     r_ch = r_ch[:, : -overlap_frame]  # Slice r_ch to remove overlap frame
# #                 else:
# #                     # Slice r_ch for overlap and pad l_ch
# #                     r_ch = r_ch[:, : -overlap_frame]  # Ensure r_ch has overlap frame removed
# #                     r_ch = torch.cat([r_ch, pause, wav], -1)  # Concatenate pause + new speech for r_ch
# #                     l_ch = torch.cat([l_ch, pause, padded], -1)  # Add silence to l_ch
# #                     l_ch = l_ch[:, : -overlap_frame]  # Slice l_ch to remove overlap frame
# #             else:
# #                 # normal concatenate with pause
# #                 padded = torch.zeros_like(wav)
# #                 if spk_id==spk_A:
# #                     l_ch = torch.cat([l_ch, pause, wav], -1)
# #                     r_ch = torch.cat([r_ch, pause, padded], -1)
# #                 else:
# #                     r_ch = torch.cat([r_ch, pause, wav], -1)
# #                     l_ch = torch.cat([l_ch, pause, padded], -1)
# 
# 
# #         full_dialog = torch.cat([l_ch, r_ch], dim=0)
# 
# #     torchaudio.save(output, full_dialog, cosyvoice.sample_rate)
# 
# 
# 
# # def XTTS_gen(script,output):
# #     sample_rate = 22050
# #     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
# 
# #     male_speakers = ['Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Xavier Hayasaka', 'Luis Moray'] #Ferran_simen, Ludvig Milivoj, Marcos Rudaski
# 
# #     female_speakers = ['Alison Dietlinde', 'Alexandra Hisakawa', 'Alma María', 'Ana Florence', 'Asya Anara',  'Andrew Chipper', 'Annmarie Nele', 'Badr Odhiambo', 'Barbora MacLean', 'Brenda Stern', 'Camilla Holmström', 'Chandra MacFarland', 'Claribel Dervla', 'Daisy Studious', 'Gitta Nikolina', 'Gracie Wise', 'Henriette Usha', 'Lidiya Szekeres', 'Lilya Stainthorpe', 'Maja Ruoho', 'Nova Hogarth', 'Narelle Moon', 'Rosemary Okafor', 'Sofia Hellen', 'Szofi Granger', 'Suad Qasim', 'Tammie Ema', 'Tammy Grit', 'Tanja Adelina', 'Uta Obando', 'Vjollca Johnnie'] #, 'Zofija Kendrick'
# 
# #     all_speakers = male_speakers + female_speakers
# #     spk_A = random.choice(all_speakers)
# #     while True:
# #         spk_B = random.choice(all_speakers)
# #         if spk_B != spk_A:
# #             break
# 
# #     for idx, (role, text) in enumerate(script, 1):
# #         spk_id = spk_A if role in ("User", "[overlap] User") else spk_B
# 
# #         wav = tts.tts(
# #             text=text,
# #             speaker=spk_id,
# #             language="en",
# #         )
# 
# #         if wav.ndim == 1:
# #             wav = wav.unsqueeze(0)
# 
# #         if idx == 1:
# #             # first utterance
# #             l_ch = wav
# #             r_ch = torch.zeros_like(wav)
# #         else:
# #             if role.strip().lower().startswith("[overlap]"):
# #                 overlap_frame = int(random.uniform(0.6, 1) * sample_rate)
# #                 padded = torch.zeros(1, wav.shape[-1] - overlap_frame)
# #                 if spk_id==spk_A:
# #                     l_ch = l_ch[ : , : - overlap_frame ]
# #                     l_ch = torch.cat([l_ch, pause, wav], -1)
# #                     r_ch = torch.cat([r_ch, pause, padded], -1)
# #                 else:
# #                     r_ch = l_ch[ : , : - overlap_frame ]
# #                     r_ch = torch.cat([l_ch, pause, wav], -1)
# #                     l_ch = torch.cat([r_ch, pause, padded], -1)
# #             else:
# #                 # normal concatenate with pause
# #                 padded = torch.zeros_like(wav)
# #                 pause = torch.zeros(1, sample_rate // 4)
# #                 if spk_id==spk_A:
# #                     l_ch = torch.cat([l_ch, pause, wav], -1)
# #                     r_ch = torch.cat([r_ch, pause, padded], -1)
# #                 else:
# #                     r_ch = torch.cat([r_ch, pause, wav], -1)
# #                     l_ch = torch.cat([l_ch, pause, padded], -1)
# 
# #             full_dialog = torch.cat([l_ch, r_ch], dim=0)
# 
# #     torchaudio.save(output, full_dialog, sample_rate)
# #     torchaudio.save("l_ch.wav", l_ch, sample_rate)
# #     torchaudio.save("r_ch.wav", r_ch, sample_rate)
# 
# 
# 
# # ──────────────────────────────  LLM UTILS  ────────────────────────────────
# # def chat_completion(model_name: str, messages: List[Dict], **gen_kwargs) -> str:
#     def encode_json(data):
#         body = json.dumps(data, ensure_ascii=False).encode("utf-8")
#         headers = {
#             "Content-Length": str(len(body)),
#             "Content-Type": "application/json; charset=utf-8",
#         }
#         return headers, httpx._content.ByteStream(body)


#     load_dotenv()
#     httpx._content.encode_json = encode_json

#     # Update these values with your own
#     api_key = os.getenv("API_KEY")
#     user_id = os.getenv("USER_ID")
#     endpoint_url = os.getenv("ENDPOINT_URL")
#     model = model_name

#     # Set the environment variable for the API key
#     os.environ["AZURE_OPENAI_KEY"] = api_key

#     # Initialize the HTTP client with SSL verification disabled
#     http_client = httpx.Client(verify=False)

#     # Initialize the AzureOpenAI client with the custom HTTP client
#     if model_name == 'aide-gpt-4o':
#         client = AzureOpenAI(
#             azure_endpoint=endpoint_url,
#             api_key=api_key,
#             api_version="2024-05-01-preview",
#             http_client=http_client,
#         )
#         response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         extra_headers={"X-User-Id": user_id},
#         max_tokens=4096,
#         )
#         return response.choices[0].message.content

#     elif model_name=='llama3.3-70b-instruct':
#         client = OpenAI(
#             api_key=api_key,
#             base_url=f"{endpoint_url}/llm/v3/models",
#             http_client=http_client,
#         )

#         extra_headers = {"x-user-id": user_id} if user_id else {}
#         response = client.chat.completions.create(
#                 model=model_name,
#                 messages=messages,
#                 extra_headers=extra_headers,
#                 n=2,
#             )
#         return response.choices[0].message.content

def chat_completion(model_name: str, messages: List[Dict], **gen_kwargs) -> str:
    """
    Unified chat completion function that works with OpenRouter API.
    OpenRouter model names format: provider/model-name
    Examples: openai/gpt-4o, meta-llama/llama-3.3-70b-instruct, anthropic/claude-3.5-sonnet
    """
    api_key = APIKEY
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Please set it in your .env file")

    # Always use OpenRouter base URL when OPENROUTER_API_KEY is set
    base_url = "https://openrouter.ai/api/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # OpenRouter requires the full model name with provider prefix (e.g., "openai/gpt-4o")
    # Do NOT remove the prefix
    completion = client.chat.completions.create(
        model=model_name,  # Keep full name like "openai/gpt-4o" or "meta-llama/llama-3.3-70b-instruct"
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        **gen_kwargs,
    )
    return completion.choices[0].message.content.strip()


# ─────────────────────────────  SCENARIO GEN  ──────────────────────────────
def generate_scenarios(cfg):
    out_path = Path(cfg.scenario["out_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 將 {n} 套入 prompt
    system_prompt = cfg.scenario["prompt"].format(n=cfg.scenario["n"])
    msgs = [{"role": "system", "content": system_prompt}]

    text = chat_completion(
        cfg.scenario["model"],
        msgs,
        # response_format={"type": "json_object"},
        # max_tokens=2048,
        # temperature=0.9,
        # top_p=0.9,
        # n=cfg.scenario["n"],
    )

    print(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        text_clean = re.sub(r"```(?:json)?|```", "", text, flags=re.I).strip()
        data = json.loads(text_clean)

    # 4️⃣ 存成 .json（一次寫完整 JSON 結構）
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


    logging.info("Saved %s → %s", type(data).__name__, out_path)

# ─────────────────────────────  DIALOGUE GEN V2 (DUAL LLM)  ──────────────────────────────

def generate_dialogues_v2(cfg):
    """
    VERSION 2: Uses TWO separate LLMs to create dialogue through interaction.
    - user_model: Generates user utterances (customer/user role)
    - agent_model: Generates agent responses (support/assistant role)
    - They alternate turns to create a natural conversation
    """
    # Read from the .jsonl file (not .json)
    scen_path = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    out_dir   = Path(cfg.dialogue["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load scenarios from JSONL format (one JSON object per line)
    scenarios = [json.loads(l) for l in scen_path.read_text(encoding="utf‑8").splitlines()]

    # Extract configuration
    user_model = cfg.dialogue["user_model"]
    agent_model = cfg.dialogue["agent_model"]
    user_system_prompt = cfg.dialogue["user_prompt"]
    agent_system_prompt = cfg.dialogue["agent_prompt"]
    max_turns = cfg.dialogue.get("max_turns", 10)
    min_turns = cfg.dialogue.get("min_turns", 4)  # Minimum turns for variety
    dialogues_per_scenario = cfg.dialogue.get("per_scenario", 3)

    logging.info(f"Generating dialogues with dual LLM system:")
    logging.info(f"  User Model: {user_model}")
    logging.info(f"  Agent Model: {agent_model}")
    logging.info(f"  Turn Range: {min_turns}-{max_turns} (randomized per dialogue)")
    logging.info(f"  Dialogues per Scenario: {dialogues_per_scenario}")

    for scenario in tqdm(scenarios, desc="dialogues"):
        scenario_desc = json.dumps(scenario, ensure_ascii=False)

        # Generate multiple dialogues for each scenario
        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_turns = []
            conversation_history = []

            # Context for both LLMs
            context_info = f"Scenario: {scenario_desc}\n\n"

            # Randomize number of turns for this dialogue to create variety
            num_turns = random.randint(min_turns, max_turns)
            logging.info(f"  Generating {scenario['id']}_{dialogue_idx + 1} with {num_turns} turns")

            # Generate dialogue turn by turn
            for turn in range(num_turns):
                if turn % 2 == 0:
                    # USER's turn
                    if turn == 0:
                        # First turn - user initiates
                        user_instruction = (
                            f"{context_info}"
                            f"This is the start of the conversation. "
                            f"Generate the user's first message based on the scenario. "
                            f"Be natural and conversational. Only output the user's utterance, nothing else."
                        )
                    else:
                        # Subsequent user turns
                        history_text = "\n".join(conversation_history)
                        user_instruction = (
                            f"{context_info}"
                            f"Conversation so far:\n{history_text}\n\n"
                            f"Generate the user's next response based on the agent's last message. "
                            f"Be natural and conversational. Only output the user's utterance, nothing else."
                        )

                    # Call user LLM
                    user_messages = [
                        {"role": "system", "content": user_system_prompt},
                        {"role": "user", "content": user_instruction}
                    ]

                    # Retry logic with exponential backoff
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

                            # Clean up if LLM adds "User:" prefix
                            user_utterance = re.sub(r'^User:\s*', '', user_utterance, flags=re.IGNORECASE)

                            if user_utterance:  # Success!
                                break
                            else:
                                logging.warning(f"Empty response from user LLM at turn {turn}, retry {retry + 1}/{max_retries}")

                        except Exception as e:
                            logging.error(f"Error in user LLM turn {turn} (retry {retry + 1}/{max_retries}): {e}")
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
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
                    # AGENT's turn
                    history_text = "\n".join(conversation_history)
                    agent_instruction = (
                        f"{context_info}"
                        f"Conversation so far:\n{history_text}\n\n"
                        f"Generate the agent's response to the user's last message. "
                        f"Be helpful, professional, and natural. Only output the agent's utterance, nothing else."
                    )

                    # Call agent LLM
                    agent_messages = [
                        {"role": "system", "content": agent_system_prompt},
                        {"role": "user", "content": agent_instruction}
                    ]

                    # Retry logic with exponential backoff
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

                            if agent_utterance:  # Success!
                                break
                            else:
                                logging.warning(f"Empty response from agent LLM at turn {turn}, retry {retry + 1}/{max_retries}")

                        except Exception as e:
                            logging.error(f"Error in agent LLM turn {turn} (retry {retry + 1}/{max_retries}): {e}")
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
                                logging.info(f"Waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                            else:
                                logging.error(f"Failed after {max_retries} retries, breaking dialogue...")

                    if agent_utterance:
                        dialogue_turns.append(f"Agent: {agent_utterance}")
                        conversation_history.append(f"Agent: {agent_utterance}")
                    else:
                        logging.error(f"Could not generate agent response after {max_retries} retries, ending dialogue early")
                        break

            # Save dialogue
            if dialogue_turns:
                dialogue_text = "\n".join(dialogue_turns)
                output_file = out_dir / f"{scenario['id']}_{dialogue_idx + 1}.txt"
                output_file.write_text(dialogue_text, encoding="utf‑8")
                logging.info(f"Saved dialogue: {output_file}")
                print(f"\n=== Generated Dialogue {dialogue_idx + 1} for {scenario['id']} ===")
                print(dialogue_text[:500] + "..." if len(dialogue_text) > 500 else dialogue_text)
                print("=" * 60)

# ────────────────────────────  OVERLAP INSERT  ─────────────────────────────

PAT_BREAK = re.compile(r"([.!?…]+)(\s|$)")


def insert_overlap(cfg):
    src = Path(cfg.dialogue["out_dir"])
    dst = Path(cfg.overlap["out_dir"]); dst.mkdir(parents=True, exist_ok=True)
    files = list(src.glob("*.txt"))

    rule_prompt = cfg.overlap.prompt

    for f in tqdm(files, desc="overlap"):
        txt = f.read_text("utf‑8")
        messages = [
            {"role": "system", "content": rule_prompt},
            {"role": "user",   "content": txt}
        ]
        new_txt = chat_completion(cfg.overlap.model, messages, max_tokens=1024, temperature=0.9)
        (dst / f.name).write_text(new_txt, "utf‑8")

# ────────────────────────────────  Filler  ────────────────────────────────────

def insert_filler(cfg):
    src = Path(cfg.overlap["out_dir"])
    dst = Path(cfg.filler["out_dir"]); dst.mkdir(parents=True, exist_ok=True)
    files = list(src.glob("*.txt"))
    for f in tqdm(files, desc="filler"):
        txt = f.read_text("utf‑8")
        messages = [
            {"role": "system", "content": cfg.filler.prompt},
            {"role": "user",   "content": txt}
        ]
        new_txt = chat_completion(cfg.filler.model, messages, max_tokens=1024, temperature=0.9)
        new_txt = format_headers_in_lines(new_txt)
        lines = new_txt.splitlines()
        new_txt = merge_overlapping_user_lines(lines)
        new_txt = "\n".join(new_txt)
        (dst / f.name).write_text(new_txt, "utf‑8")

# ────────────────────────────────  Judge  ────────────────────────────────────

def llm_judge(cfg):
    src = Path(cfg.filler["out_dir"])
    dst = Path(cfg.judge["out_dir"]); dst.mkdir(parents=True, exist_ok=True)
    for prefix in range(1, cfg.scenario.n + 1):
        group_files = sorted(src.glob(f"scenario{prefix}_*.txt"))
        results = []
        for f in tqdm(group_files, desc=f"Processing scenario{prefix}"):
            txt = f.read_text(encoding="utf-8")
            messages = [
                {"role": "system", "content": cfg.judge.prompt},
                {"role": "user",   "content": txt}
            ]
            response = chat_completion(cfg.filler.model, messages, max_tokens=1024, temperature=0.9)
            # Extract score from response
            print(response)
            score = int(re.search(r"Score:\s*(\d+)", response).group(1))

            results.append({"file": f, "score": score})
        # Sort and save top X
        top_x = cfg.judge.top_x
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        top_results = results_sorted[:top_x]
        new_folder = dst / f"top_{top_x}_scenario{prefix}"
        new_folder.mkdir(parents=True, exist_ok=True)
        for item in top_results:
            src_file = item["file"]
            dst_file = new_folder / src_file.name
            dst_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

# ────────────────────────────────  TTS  ────────────────────────────────────

def tts_batch(cfg):
    mode = cfg.tts.mode
    src = Path(cfg.tts.load_dir)
    wav_dir = Path(cfg.tts.wav_dir); wav_dir.mkdir(parents=True, exist_ok=True)

    # Create error log file
    error_log_path = wav_dir / "tts_errors.txt"

    # Process folder by folder (data1, data2, etc.)
    for data_folder in sorted([f for f in src.iterdir() if f.is_dir()]):
        # Look for dialogue_multi_txt subfolder
        if True: #data_folder.name in ["data12-2"]:
        #     continue
        # else:
            txt_folder = data_folder / "dialogue_multi_txt"
            if not txt_folder.exists():
                print(f"Warning: {data_folder.name}/dialogue_multi_txt not found, skipping...")
                continue

            # Create corresponding output folder structure
            folder_wav_dir = wav_dir / data_folder.name / "dialogue_multi_txt"
            folder_wav_dir.mkdir(parents=True, exist_ok=True)

            for txt_file in tqdm(list(txt_folder.glob("*.txt")), desc=f"tts-{data_folder.name}"):
                if cfg.tts.model == "CosyVoice":
                    post_fix = cfg.tts["postfix"]
                    wav_file_path = folder_wav_dir / f"{txt_file.stem}.wav"
                elif cfg.tts.model == "XTTS":
                    wav_file_path = folder_wav_dir / f"{txt_file.stem}_XTTS.wav"
                else:
                    wav_file_path = folder_wav_dir / f"{txt_file.stem}.wav"

                if wav_file_path.exists():
                    print(f"Skipping {data_folder.name}/dialogue_multi_txt/{txt_file.name} - wav file already exists")
                    continue
                try:
                    text = txt_file.read_text("utf-8")
                    script = []
                    for idx, line in enumerate(text.strip().splitlines()):
                        line = line.strip()
                        if not line:
                            continue
                        if ":" in line:
                            role, content = line.split(":", 1)
                            script.append((role.strip(), content.strip()))
                    print(script)
                    if cfg.tts.model == "CosyVoice":
                        post_fix = cfg.tts["postfix"]
                        CosyVoice_gen(mode, script, folder_wav_dir / f"{txt_file.stem}.wav")
                    if cfg.tts.model == "XTTS":
                        XTTS_gen(script, folder_wav_dir / f"{txt_file.stem}_XTTS.wav")
                except Exception as e:
                    # Log the error and continue with next file
                    print(f"Error processing {data_folder.name}/dialogue_multi_txt/{txt_file.name}: {str(e)}")
                    with open(error_log_path, "a", encoding="utf-8") as error_file:
                        error_file.write(f"{data_folder.name}/dialogue_multi_txt/{txt_file.name}: {str(e)}\n")
                    continue

## ──────────────────────────  Control  ───────────────────────────────

def control_dialogues(cfg):
    out_path = Path(cfg.control["out_dir"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)


    # 將 {n} 套入 prompt
    user_prompt = cfg.control["prompt"]


    for s in tqdm(range(cfg.control["n"]), desc="dialogues"):

        messages = [
            {"role": "system", "content": "you are a good instructions following model."},
            {"role": "user",   "content": user_prompt}
        ]
        script = chat_completion(cfg.control["model"], messages,
                                max_tokens=1024,
                                temperature=0.9, top_p=0.9)
        print(script)
        path = out_path / f"control_{s}.txt"
        path.write_text(script, encoding="utf‑8")

## ──────────────────────────  Human ───────────────────────────────

def laughter_breath_dialogues(cfg):
    out_path = Path(cfg.human["out_dir"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)


    # 將 {n} 套入 prompt
    user_prompt = cfg.human["prompt"]


    for s in tqdm(range(cfg.human["n"]), desc="dialogues"):

        messages = [
            {"role": "system", "content": "you are a good instructions following model."},
            {"role": "user",   "content": user_prompt}
        ]
        script = chat_completion(cfg.human["model"], messages,
                                max_tokens=1024,
                                temperature=0.9, top_p=0.9)
        print(script)
        path = out_path / f"human_{s}.txt"
        path.write_text(script, encoding="utf‑8")

# ─────────────────────────────  ORCHESTRATOR  ──────────────────────────────

@dataclass
class PipelineConfig:
    # device
    device: str = "cuda"
    # stages to run
    stages: List[str] = field(default_factory=lambda: [
        "scenario", "dialogue", "overlap", "tts"])

    # scenario
    scenario: OmegaConf = OmegaConf.create({
        "model": "gpt-4o-mini",
        "n": 200,
        "out_file": "data/scenarios.jsonl",
    })

    # dialogue (VERSION 2 - Dual LLM)
    dialogue: OmegaConf = OmegaConf.create({
        "user_model": "llama3.3-70b-instruct",  # Model for user role
        "agent_model": "aide-gpt-4o",  # Model for agent role
        "user_prompt": "You are a customer/user interacting with a support agent or service. Be natural, ask questions, express concerns, and respond realistically based on the conversation context.",
        "agent_prompt": "You are a helpful customer support agent or virtual assistant. Provide clear, professional, and helpful responses to the user's questions and concerns.",
        "out_dir": "data/dialog_txt_v2",
        "min_turns": 4,  # Minimum turns per dialogue (for variety)
        "max_turns": 10,  # Maximum turns per dialogue (randomized)
        "per_scenario": 3,  # Number of dialogues to generate per scenario
    })

    # overlap
    overlap: OmegaConf = OmegaConf.create({
        "model": "gpt-4o-mini",
        "out_dir": "data/overlap_txt",
    })

    # filler
    filler: OmegaConf = OmegaConf.create({
        "model": "gpt-4o-mini",
        "out_dir": "data/filler_txt",
    })

    judge: OmegaConf = OmegaConf.create({
        "model": "gpt-4o-mini",
        "out_dir": "data/filler_txt",
    })

    # tts
    tts: OmegaConf = OmegaConf.create({
        "model": "CosyVoice2-0.5B",
        "spk_bank": "resources/speakers.json",
        "wav_dir": "data/wav",
    })

    control: OmegaConf = OmegaConf.create({
        "model": "gpt-4o-mini",
        "n": 200,
        "out_dir": "data/scenarios.jsonl",
    })

    human: OmegaConf = OmegaConf.create({
        "model": "gpt-4o-mini",
        "n": 200,
        "out_dir": "data/scenarios.jsonl",
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
            convert_nested_json_to_jsonl(original_path,original_path.with_suffix(".jsonl"))
        if "dialogue" in st:
            print("Dialogue Generating (V2 - Dual LLM)...")
            generate_dialogues_v2(self.cfg)  # Use V2 function with dual LLM
        if "overlap" in st:
            print("Inserting overlap dialogue...")
            insert_overlap(self.cfg)
        if "filler" in st:
            print("Inserting filler dialogue...")
            insert_filler(self.cfg)
        if "judge" in st:
            print("Judge dialogue...")
            llm_judge(self.cfg)
        # if "control" in st:
        #     print("Generate control dialogue...")
        #     control_dialogues(self.cfg)
        # if "human" in st:
            # print("Generate humanity dialogue...")
            # insert_overlap(self.cfg)
        if "tts" in st:
            print("Speech dialogue Generating...")
            tts_batch(self.cfg)


# ─────────────────────────────  CLI ENTRY  ────────────────────────────────

def main():  # pragma: no cover
    @hydra.main(config_path="conf", config_name="base_v2", version_base=None)
    def _run(cfg):
        # 不要再 to_object
        # cfg = OmegaConf.to_object(cfg)

        # 如果要轉 PipelineConfig dataclass
        if not isinstance(cfg, PipelineConfig):
            # OmegaConf 直接轉 dataclass
            cfg = OmegaConf.merge(OmegaConf.structured(PipelineConfig), cfg)

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        Pipeline(cfg).run()

    print("Finish Config Matching")
    _run()



if __name__ == "__main__":
    main()

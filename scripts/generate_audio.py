import os
import sys
import argparse
import threading
import uuid
import json
from contextlib import nullcontext
from hyperpyyaml import load_hyperpyyaml

from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
import torch
import torchaudio
import numpy as np
from torch.nn import functional as F

from taste_speech.taste2.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from taste_speech.taste2.cosyvoice.utils.file_utils import logging
from taste_speech.taste2.cosyvoice.utils.common import fade_in_out
from taste_speech.taste2.taste2_interface import TASTE2Model
from taste_speech.taste2.taste_sft_utils import apply_template_on_message


def print_green(text):
    """Print text in green color"""
    print(f"\033[92m{text}\033[0m")


def filter_structural_tokens(text_tokens, taste_embs, tokenizer):
    """
    Filter out structural tokens (im_start, im_end, role, newlines at boundaries)
    to get only content tokens for audio generation and clean text.

    Returns:
        filtered_tokens: Content tokens only
        filtered_embs: Taste embeddings for content tokens only
        clean_text: Decoded text without structural tokens
    """
    # Get special token IDs
    im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    # Get role token IDs (assistant, user, system)
    role_ids = set()
    for role in ["assistant", "user", "system"]:
        role_ids.update(tokenizer.encode(role, add_special_tokens=False))

    # Filter tokens
    tokens_list = text_tokens[0].tolist() if text_tokens.dim() > 1 else text_tokens.tolist()

    # State machine to track position in ChatML format
    filtered_indices = []
    i = 0
    while i < len(tokens_list):
        token = tokens_list[i]

        # Skip im_start
        if token == im_start_id:
            i += 1
            continue

        # Skip im_end
        if token == im_end_id:
            i += 1
            continue

        # Skip role tokens
        if token in role_ids:
            i += 1
            continue

        # Skip newline after role (first newline after im_start+role)
        if token == newline_id and i > 0:
            # Check if previous tokens were im_start or role
            if i >= 2:
                prev_token = tokens_list[i-1]
                if prev_token in role_ids or prev_token == im_start_id:
                    i += 1
                    continue

        # Keep this token (it's content)
        filtered_indices.append(i)
        i += 1

    # Extract filtered tokens and embeddings
    if filtered_indices:
        filtered_tokens = torch.tensor([tokens_list[i] for i in filtered_indices],
                                      dtype=text_tokens.dtype).unsqueeze(0)
        filtered_embs = taste_embs[:, filtered_indices, :] if taste_embs.dim() == 3 else taste_embs[filtered_indices, :]
        if filtered_embs.dim() == 2:
            filtered_embs = filtered_embs.unsqueeze(0)

        # Decode clean text
        clean_text = tokenizer.decode([tokens_list[i] for i in filtered_indices])
    else:
        # No content tokens found
        filtered_tokens = torch.zeros(1, 0, dtype=text_tokens.dtype, device=text_tokens.device)
        filtered_embs = torch.zeros(1, 0, taste_embs.size(-1), dtype=taste_embs.dtype, device=taste_embs.device)
        clean_text = ""

    return filtered_tokens, filtered_embs, clean_text

def load_audio_data(test_file, output_dir):
    """Unified function to load audio from either arrow file or audio file"""
    assert os.path.exists(test_file)
    
    if test_file.endswith('.arrow'):
        from datasets import Dataset
        print("Loading dataset sample...")
        dataset = Dataset.from_file(test_file)
        sample = dataset[0]
        
        audio_array = sample['mp3']['array']
        sr = sample['mp3']['sampling_rate']
        audio_16k = torch.tensor(audio_array, dtype=torch.float32)
        ground_truth_text = sample['json']['text']
        audio_basename = "audio"
    else:
        print(f"Loading audio file: {test_file}")
        audio, sr = torchaudio.load(test_file)
        audio_16k = audio
        ground_truth_text = None
        audio_basename = os.path.basename(test_file).split('.')[0]
    
    # Convert to mono and resample to 16kHz
    if audio_16k.dim() > 1:
        audio_16k = audio_16k.mean(dim=0, keepdim=True)
    else:
        audio_16k = audio_16k.unsqueeze(0)
    
    if sr != 16000:
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_16k)
    
    print(f"Audio shape: {audio_16k.shape}, Duration: {audio_16k.shape[-1] / 16000:.2f}s")
    
    # Save original audio
    ext = ".mp3" if test_file.endswith('.arrow') else ".wav"
    original_path = os.path.join(output_dir, f"{audio_basename}_original{ext}")
    torchaudio.save(original_path, audio_16k, 16000)
    print(f"Saved original audio to: {original_path}")
    
    return audio_16k, ground_truth_text, audio_basename


class TASTE2:
    """Unified TASTE2 class for both Stage 1 and Stage 2 audio reconstruction"""
    
    def __init__(self, model_dir, stage=1, load_jit=False, load_trt=False, load_trt_llm=False, load_vllm=False, fp16=False, trt_concurrent=1):
        # Simply instantiate the complete TASTE2Model
        self.model = TASTE2Model(
            model_dir=model_dir,
            stage=stage,
            load_jit=load_jit,
            load_trt=load_trt,
            load_trt_llm=load_trt_llm,
            load_vllm=load_vllm,
            fp16=fp16,
            trt_concurrent=trt_concurrent,
        )

        # Expose common references for backward compatibility
        self.device = self.model.device
        self.frontend = self.model.frontend
        self.audio_extractor = self.model.audio_extractor
        self.sample_rate = self.model.sample_rate
        self.taste_stage1 = self.model.taste_stage1
        self.stage = self.model.stage
        self.fp16 = self.model.fp16

    def _run_asr(self, audio_16k, asr_model_dir="openai/whisper-large-v3"):
        """Run ASR on audio to get text transcription"""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(asr_model_dir)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )
        
        return pipe(audio_16k.squeeze().numpy())["text"]

    def _preprocess(self, audio_16k, asr_model_dir=None, text=None):
        assert asr_model_dir or text, "Either ASR model or text must be provided"

        # Get text from ASR or use provided text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        
        # Extract text and audio tokens
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        # Use audio_extractor directly like in generate_stage1.py
        audio_feature, audio_feature_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]])
        
        # Convert to half precision if model is using fp16
        if self.fp16:
            audio_feature = audio_feature.half()
        
        # Move to correct device
        audio_feature = audio_feature.to(self.device)

        taste_tokenizer = self.taste_stage1.taste_tokenizer
        tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']

        return dict(
            text_token=text_token.to(self.device),
            text_token_len=text_token_len.to(self.device),
            taste_token_emb=taste_token_emb.to(self.device),
        )

    def _sft_preprocess(self, audio_16k, text, role="user", system_prompt="You are a helpful assistant.", asr_model_dir=None):
        """
        SFT preprocessing function that formats text with special tokens
        similar to format_and_concatenate_conversation function

        Builds the full ChatML prompt:
            <|im_start|>system\n{system_prompt}<|im_end|>
            <|im_start|>user\n{text}<|im_end|>
            <|im_start|>assistant\n

        Args:
            audio_16k: Input audio tensor
            text: Text string for the conversation turn
            role: Role ('system', 'user', 'assistant') for this turn
            system_prompt: System prompt text
            asr_model_dir: Optional ASR model for fallback

        Returns:
            dict with formatted conversation tokens and features
        """
        assert asr_model_dir or text, "Either ASR model or text must be provided"

        # Get text from ASR or use provided text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)

        tokenizer = self.frontend.tokenizer

        # Build system message: <|im_start|>system\n{system_prompt}<|im_end|>
        system_ids = (
            tokenizer.encode("<|im_start|>", add_special_tokens=False) +
            tokenizer.encode("system", add_special_tokens=False) +
            tokenizer.encode("\n", add_special_tokens=False) +
            tokenizer.encode(system_prompt, add_special_tokens=False) +
            tokenizer.encode("<|im_end|>", add_special_tokens=False)
        )
        system_message_ids = [-1] * len(system_ids)  # all structural (no audio)

        # Build user message: <|im_start|>user\n{text}<|im_end|>
        formatted_text_token, formatted_text_token_len, token_message_ids = apply_template_on_message(tokenizer, role, asr_text)

        # Build assistant prefix: <|im_start|>assistant\n
        assistant_prefix_ids = (
            tokenizer.encode("<|im_start|>", add_special_tokens=False) +
            tokenizer.encode("assistant", add_special_tokens=False) +
            tokenizer.encode("\n", add_special_tokens=False)
        )
        assistant_message_ids = [-1] * len(assistant_prefix_ids)  # structural

        # Concatenate: system + user + assistant prefix
        system_tensor = torch.tensor(system_ids, dtype=formatted_text_token.dtype).unsqueeze(0)
        assistant_tensor = torch.tensor(assistant_prefix_ids, dtype=formatted_text_token.dtype).unsqueeze(0)
        formatted_text_token = torch.cat([system_tensor, formatted_text_token, assistant_tensor], dim=1)
        formatted_text_token_len = torch.tensor([formatted_text_token.shape[1]], dtype=torch.int32)
        token_message_ids = system_message_ids + token_message_ids + assistant_message_ids

        print(formatted_text_token)
        print(token_message_ids)

        # Extract audio features similar to _preprocess
        audio_feature, audio_feature_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]])

        # Convert to half precision if model is using fp16
        if self.fp16:
            audio_feature = audio_feature.half()

        # Move to correct device
        audio_feature = audio_feature.to(self.device)

        # Use taste_tokenizer similar to _preprocess
        taste_tokenizer = self.taste_stage1.taste_tokenizer
        tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']

        return dict(
            text_token=text_token.to(self.device),
            text_token_len=text_token_len.to(self.device),
            taste_token_emb=taste_token_emb.to(self.device),
            formatted_text_token=formatted_text_token.to(self.device),
            formatted_text_token_len=formatted_text_token_len.to(self.device),
            token_message_ids=torch.tensor(token_message_ids).to(self.device),
        )

    def _postprocess(self, s3_tokens, audio_16k):
        if s3_tokens.shape[1] == 0:
            raise ValueError("No speech tokens generated - cannot synthesize audio. "
                             "The model likely produced no meaningful content tokens.")
        # Convert tokens back to audio
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        audio_chunks = list(self.model.tts(
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding
        ))
        
        return torch.cat([chunk['tts_speech'] for chunk in audio_chunks], dim=1)

    @torch.inference_mode()
    def reconstruct_stage1(self, audio_16k, asr_model_dir=None, text=None):
        """Main reconstruction pipeline for Stage 1"""
        if self.stage != 1:
            raise ValueError("reconstruct_stage1 can only be called on stage 1 model")
        
        # Get ASR text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        
        data = self._preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)

        # Generate speech tokens from text and audio
        s3_tokens = list(self.taste_stage1.inference(
            **data,
            sampling=25
        ))
        s3_tokens = torch.tensor(s3_tokens, dtype=torch.long).unsqueeze(0) if s3_tokens else torch.zeros(1, 0, dtype=torch.long)
        
        output_audio = self._postprocess(s3_tokens, audio_16k)
        
        # Return results with text info
        return {
            'audio': output_audio,
            'asr_text': asr_text
        }

    @torch.inference_mode()
    def generation_stage2(self, audio_16k, asr_model_dir=None, text=None, min_len=5, max_len=100, quantize_input_latent=False):
        """Generate stage 2 output"""
        if self.stage != 2:
            raise ValueError("generation_stage2 can only be called on stage 2 model")

        # Get ASR text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)

        data = self._preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)

        slm_output_generator = self.model.slm.inference(
            **data,
            min_len=min_len,
            max_len=max_len,
            sampling=25,
            stop_id=self.model.slm.eos_token_id,
            quantize_input_latent=quantize_input_latent,
        )
        
        new_text_tokens, new_taste_embs = list(), list()
        for new_text_token, new_taste_emb in slm_output_generator:
            new_text_tokens.append(new_text_token)
            new_taste_embs.append(new_taste_emb.squeeze())

        # Combine original and new tokens
        if new_text_tokens:
            # Convert new tokens to tensors
            new_text_tokens_tensor = torch.cat(new_text_tokens, dim=0).unsqueeze(0) if new_text_tokens else torch.zeros(1, 0, dtype=torch.int32)
            new_taste_embs_tensor = torch.stack(new_taste_embs).unsqueeze(0) if new_taste_embs else torch.zeros(1, 0, 192)
            
            # Concatenate original + new tokens
            final_text_token = torch.cat([data['text_token'], new_text_tokens_tensor.to(self.device)], dim=1)
            final_text_token_len = torch.tensor([final_text_token.shape[1]], dtype=torch.int32).to(self.device)
            final_taste_token_emb = torch.cat([data['taste_token_emb'], new_taste_embs_tensor.to(self.device)], dim=1)
        else:
            # Fallback to original tokens if no new tokens generated
            final_text_token = data['text_token']
            final_text_token_len = data['text_token_len']
            final_taste_token_emb = data['taste_token_emb']

        # Decode the generated text
        generated_text = self.frontend.tokenizer.decode(final_text_token[0].tolist())
        print_green('Completion: {}'.format(generated_text))

        s3_tokens = list(self.taste_stage1.inference(
            text_token=final_text_token,
            text_token_len=final_text_token_len,
            taste_token_emb=final_taste_token_emb,
            sampling=25,
        ))
        s3_tokens = torch.tensor(s3_tokens, dtype=torch.long).unsqueeze(0) if s3_tokens else torch.zeros(1, 0, dtype=torch.long)

        output_audio = self._postprocess(s3_tokens, audio_16k)
        
        # Return results with text info
        return {
            'audio': output_audio,
            'asr_text': asr_text,
            'generated_text': generated_text
        }
        
    @torch.inference_mode()
    def generation_stagesft(self, audio_16k, asr_model_dir=None, text=None, min_len=1, max_len=100):
        """Generate stage 2 output"""
        if self.stage != 'sft':
            raise ValueError("generation_stagesft can only be called on stage sft model")

        # Get ASR text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)

        data = self._sft_preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)

        stop_id = self.frontend.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        slm_output_generator = self.model.slm.inference(
            **data,
            min_len=min_len,
            max_len=max_len,
            sampling=25,
            stop_id=stop_id
        )

        new_text_tokens, new_taste_embs = list(), list()
        for new_text_token, new_taste_emb in slm_output_generator:
            new_text_tokens.append(new_text_token)
            new_taste_embs.append(new_taste_emb.squeeze())

        # Combine original and new tokens
        if new_text_tokens:
            # Convert new tokens to tensors
            new_text_tokens_tensor = torch.cat(new_text_tokens, dim=0).unsqueeze(0) if new_text_tokens else torch.zeros(1, 0, dtype=torch.int32)
            new_taste_embs_tensor = torch.stack(new_taste_embs).unsqueeze(0) if new_taste_embs else torch.zeros(1, 0, 192)

            # Concatenate original + new tokens
            final_text_token = new_text_tokens_tensor.to(self.device)
            final_text_token_len = torch.tensor([final_text_token.shape[1]], dtype=torch.int32).to(self.device)
            final_taste_token_emb = new_taste_embs_tensor.to(self.device)
        else:
            # Fallback to original tokens if no new tokens generated
            final_text_token = data['text_token']
            final_text_token_len = data['text_token_len']
            final_taste_token_emb = data['taste_token_emb']

        # Decode the generated text (with format tokens for debugging)
        generated_text_raw = self.frontend.tokenizer.decode(final_text_token[0].tolist())
        print_green('Completion (raw): {}'.format(generated_text_raw))

        # Filter out structural tokens for audio generation and clean text
        filtered_tokens, filtered_embs, clean_text = filter_structural_tokens(
            final_text_token, final_taste_token_emb, self.frontend.tokenizer
        )
        filtered_token_len = torch.tensor([filtered_tokens.shape[1]], dtype=torch.int32).to(self.device)
        print_green('Completion (clean): {}'.format(clean_text))

        # Generate audio using only content tokens
        s3_tokens = list(self.taste_stage1.inference(
            text_token=filtered_tokens.to(self.device),
            text_token_len=filtered_token_len,
            taste_token_emb=filtered_embs.to(self.device),
            sampling=25,
        ))
        s3_tokens = torch.tensor(s3_tokens, dtype=torch.long).unsqueeze(0) if s3_tokens else torch.zeros(1, 0, dtype=torch.long)

        output_audio = self._postprocess(s3_tokens, audio_16k)

        # Return results with text info
        return {
            'audio': output_audio,
            'asr_text': asr_text,
            'generated_text': clean_text  # Return clean text without format tokens
        }

def process_single_file(model, audio_file, output_dir, asr_model_dir, stage, quantize_input_latent=False):
    """Process a single audio file"""
    print(f"\nProcessing: {os.path.basename(audio_file)}")
    
    # Load audio data
    audio_16k, ground_truth_text, audio_basename = load_audio_data(audio_file, output_dir)
    
    results = {}
    
    # Run the specified stage
    if stage == '1':
        print(f"\n=== Running Stage 1 for {audio_basename} ===")
        try:
            result = model.reconstruct_stage1(
                audio_16k, asr_model_dir=asr_model_dir
            )
            
            print(f"ASR text: {result['asr_text']}")
            
            # Save Stage 1 results
            output_path = os.path.join(output_dir, f"{audio_basename}_stage1_reconstructed.mp3")
            torchaudio.save(output_path, result['audio'].cpu(), model.sample_rate, backend='soundfile')
            print(f"Stage 1 results saved to: {output_path}")
            
            results['stage1'] = {
                'audio_path': output_path,
                'asr_text': result['asr_text']
            }
            
        except Exception as e:
            print(f"Stage 1 Error for {audio_basename}: {e}")
            import traceback
            traceback.print_exc()
    
    elif stage == '2':
        print(f"\n=== Running Stage 2 for {audio_basename} ===")
        try:
            result = model.generation_stage2(
                audio_16k=audio_16k,
                asr_model_dir=asr_model_dir,
                quantize_input_latent=quantize_input_latent,
            )
            
            print(f"ASR text: {result['asr_text']}")
            print(f"Generated text: {result['generated_text']}")
            
            # Save Stage 2 results
            output_path = os.path.join(output_dir, f"{audio_basename}_stage2_reconstructed.mp3")
            torchaudio.save(output_path, result['audio'].cpu(), model.sample_rate, backend='soundfile')
            print(f"Stage 2 results saved to: {output_path}")
            
            results['stage2'] = {
                'audio_path': output_path,
                'asr_text': result['asr_text'],
                'generated_text': result['generated_text']
            }
            
        except Exception as e:
            print(f"Stage 2 Error for {audio_basename}: {e}")
            import traceback
            traceback.print_exc()

    elif stage == 'sft':
        print(f"\n=== Running Stage SFT for {audio_basename} ===")
        try:
            result = model.generation_stagesft(
                audio_16k=audio_16k,
                asr_model_dir=asr_model_dir
            )
            
            print(f"ASR text: {result['asr_text']}")
            print(f"Generated text: {result['generated_text']}")
            
            # Save Stage sft results
            output_path = os.path.join(output_dir, f"{audio_basename}_stage_sft_reconstructed.mp3")
            torchaudio.save(output_path, result['audio'].cpu(), model.sample_rate, backend='soundfile')
            print(f"Stage sft results saved to: {output_path}")
            
            results['stagesft'] = {
                'audio_path': output_path,
                'asr_text': result['asr_text'],
                'generated_text': result['generated_text']
            }
            
        except Exception as e:
            print(f"Stage SFT Error for {audio_basename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save individual file results
    json_results = {
        'original_text': ground_truth_text,
        'audio_basename': audio_basename,
        'audio_file': audio_file,
        'results': results
    }
    
    # Save individual JSON file
    json_path = os.path.join(output_dir, f"{audio_basename}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print_green(f"Results saved to: {json_path}")
    
    return json_results


def main():
    parser = argparse.ArgumentParser(description='TASTE2 Audio Generation (Stage 1 & 2)')
    parser.add_argument('--model_dir', type=str, required=True, help='TASTE2 model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--test_files', type=str, nargs='+', required=True, help='Audio/Arrow file paths (multiple files supported)')
    parser.add_argument('--asr_model_dir', type=str, default="openai/whisper-large-v3", help='ASR model')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'sft'], default='1',
                        help='Which stage to run: 1, 2 or sft (default: 1)')
    parser.add_argument('--quantize_input_latent', action='store_true', default=False,
                        help='Quantize taste latent via RVQ then dequantize before feeding back as next input (stage 2 only)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_files = args.test_files
    
    print(f"Processing {len(test_files)} audio files...")
    print(f"Output directory: {args.output_dir}")
    print(f"Stage: {args.stage}")
    
    # Initialize model once
    if args.stage == '1':
        print("\nInitializing TASTE2 model for Stage 1...")
        model = TASTE2(args.model_dir, stage=1, fp16=False)
    elif args.stage == '2':
        print("\nInitializing TASTE2 model for Stage 2...")
        model = TASTE2(args.model_dir, stage=2, fp16=False)
    elif args.stage == 'sft':
        print("\nInitializing TASTE2 model for Stage 2...")
        model = TASTE2(args.model_dir, stage='sft', fp16=False)
    
    # Process all files
    all_results = []
    successful_files = 0
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(test_files)}: {os.path.basename(test_file)}")
        print('='*60)
        
        try:
            if not os.path.exists(test_file):
                print(f"Warning: File {test_file} does not exist, skipping...")
                continue
                
            file_results = process_single_file(
                model, test_file, args.output_dir, args.asr_model_dir, args.stage,
                quantize_input_latent=args.quantize_input_latent,
            )
            all_results.append(file_results)
            successful_files += 1
            
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save batch results summary
    batch_results = {
        'total_files': len(test_files),
        'successful_files': successful_files,
        'failed_files': len(test_files) - successful_files,
        'stage': args.stage,
        'model_dir': args.model_dir,
        'asr_model_dir': args.asr_model_dir,
        'files': all_results
    }
    
    batch_json_path = os.path.join(args.output_dir, "batch_results.json")
    with open(batch_json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print_green(f"\n{'='*60}")
    print_green(f"BATCH PROCESSING COMPLETE")
    print_green(f"{'='*60}")
    print_green(f"Total files: {batch_results['total_files']}")
    print_green(f"Successful: {batch_results['successful_files']}")
    print_green(f"Failed: {batch_results['failed_files']}")
    print_green(f"Batch results saved to: {batch_json_path}")
    print_green(f"All outputs in: {args.output_dir}")

if __name__ == '__main__':
    main()
import os
import sys
import argparse
import threading
import uuid
import json
from contextlib import nullcontext
from hyperpyyaml import load_hyperpyyaml

# Add CosyVoice to Python path
cosyvoice_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CosyVoice')
if cosyvoice_path not in sys.path:
    sys.path.insert(0, cosyvoice_path)
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
import torch
import torchaudio
import numpy as np
from torch.nn import functional as F

from taste_speech.taste2.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from taste_speech.taste2.cosyvoice.utils.file_utils import logging
from taste_speech.taste2.cosyvoice.utils.common import fade_in_out

def print_green(text):
    """Print text in green color"""
    print(f"\033[92m{text}\033[0m")

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

class TASTE2Model:
    """Unified TASTE2 Model for both Stage 1 and Stage 2 audio reconstruction"""
    
    def __init__(self, llm, flow, hift, slm=None, fp16=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm, self.flow, self.hift = llm, flow, hift
        self.slm = slm  # Optional SLM for Stage 2
        self.fp16 = fp16
        
        if fp16:
            self.llm.half()
            self.flow.half()
        
        # Cache and streaming parameters
        self.token_hop_len = 25
        self.mel_cache_len = 8
        self.source_cache_len = self.mel_cache_len * 480
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        self.session_data = {}
    
    def load(self, llm_model, flow_model, hift_model, slm_model=None):
        """Load model weights"""
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
        if slm_model is not None:
            self.slm.load_state_dict(torch.load(slm_model, map_location=self.device), strict=True)
            self.slm.to(self.device).eval()
    
    def _run_inference_job(self, source_token, session_id, **kwargs):
        """Unified inference job for both LLM and VC"""
        if source_token.shape[1] == 0:  # LLM job
            with self.llm_context, torch.cuda.amp.autocast(self.fp16 and not hasattr(self.llm, 'vllm')):
                for token in self.llm.inference(**{k: v.to(self.device) if hasattr(v, 'to') else v for k, v in kwargs.items()}, uuid=session_id):
                    self.session_data[session_id]['tokens'].append(token)
        else:  # VC job
            self.session_data[session_id]['tokens'] = source_token.flatten().tolist()
        self.session_data[session_id]['finished'] = True
    
    def token2wav(self, token, prompt_token, prompt_feat, embedding, session_id, token_offset=0, finalize=True, speed=1.0):
        """Convert tokens to waveform"""
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=False,
                finalize=finalize
            )
        
        if speed != 1.0 and finalize:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))
        return tts_speech
    
    def tts(self, source_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), speed=1.0, **kwargs):
        """Simplified TTS generation for reconstruction"""
        session_id = str(uuid.uuid1())
        
        with self.lock:
            self.session_data[session_id] = {'tokens': [], 'finished': False}
        
        # Run inference job
        job_kwargs = {k: v for k, v in kwargs.items() if k in ['text', 'prompt_text', 'llm_prompt_speech_token', 'llm_embedding']}
        thread = threading.Thread(target=self._run_inference_job, args=(source_speech_token, session_id), kwargs=job_kwargs)
        thread.start()
        thread.join()
        
        # Convert tokens to audio
        tokens = torch.tensor(self.session_data[session_id]['tokens']).unsqueeze(0)
        speech = self.token2wav(
            token=tokens,
            prompt_token=kwargs.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
            prompt_feat=kwargs.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
            embedding=flow_embedding,
            session_id=session_id,
            speed=speed
        )
        
        # Cleanup
        with self.lock:
            self.session_data.pop(session_id)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield {'tts_speech': speech.cpu()}

class TASTE2:
    """Unified TASTE2 class for both Stage 1 and Stage 2 audio reconstruction"""
    
    def __init__(self, model_dir, stage=1, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dir = model_dir
        self.fp16 = fp16
        self.stage = stage
        
        # Download model if needed
        if not os.path.exists(model_dir):
            if stage == 1:
                model_dir = snapshot_download(model_dir)
            else:
                model_dir = hf_snapshot_download(model_dir)
        
        # Load appropriate config file
        config_file = f'taste2_stage{stage}.yaml'
        hyper_yaml_path = os.path.join(model_dir, config_file)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f'{hyper_yaml_path} not found!')
        
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        # Initialize frontend and extractors
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            os.path.join(model_dir, 'campplus.onnx'),
            os.path.join(model_dir, 'speech_tokenizer_v2.onnx'),
            os.path.join(model_dir, 'spk2info.pt'),
            configs['allowed_special']
        )
        self.audio_extractor = configs['audio_extractor']
        self.sample_rate = configs['sample_rate']
        
        # Handle CUDA availability
        if not torch.cuda.is_available() and (load_jit or load_trt or fp16):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        
        # Initialize model based on stage
        slm = configs.get('slm') if stage == 2 else None
        self.model = TASTE2Model(configs['llm'], configs['flow'], configs['hift'], slm, fp16)
        
        # Load model weights
        self.model.load(
            os.path.join(model_dir, 'llm.pt'),
            os.path.join(model_dir, 'flow.pt'),
            os.path.join(model_dir, 'hift.pt'),
            os.path.join(model_dir, 'slm.pt') if stage == 2 else None
        )
        
        # Optional optimizations
        if load_vllm:
            self.model.load_vllm(os.path.join(model_dir, 'vllm'))
        if load_jit:
            precision = 'fp16' if fp16 else 'fp32'
            self.model.load_jit(os.path.join(model_dir, f'flow.encoder.{precision}.zip'))
        if load_trt:
            precision = 'fp16' if fp16 else 'fp32'
            self.model.load_trt(
                os.path.join(model_dir, f'flow.decoder.estimator.{precision}.mygpu.plan'),
                os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx'),
                trt_concurrent,
                fp16
            )

        self.taste_stage1 = self.model.slm.taste_stage1 if stage == 2 else self.model.llm
        
        del configs

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

    def _postprocess(self, s3_tokens, audio_16k):
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
    def generation_stage2(self, audio_16k, asr_model_dir=None, text=None):
        """Generate stage 2 output"""
        if self.stage != 2:
            raise ValueError("generation_stage2 can only be called on stage 2 model")
        
        # Get ASR text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        
        data = self._preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)
        
        slm_output_generator = self.model.slm.inference(
            **data,
            min_len=3,
            max_len=20,
            sampling=25,
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

def process_single_file(model, audio_file, output_dir, asr_model_dir, stage):
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
                asr_model_dir=asr_model_dir
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
    parser.add_argument('--stage', type=str, choices=['1', '2'], default='1', 
                        help='Which stage to run: 1 or 2 (default: 1)')
    
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
                model, test_file, args.output_dir, args.asr_model_dir, args.stage
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
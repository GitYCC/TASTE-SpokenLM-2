
import os
import argparse
import threading
import uuid
from contextlib import nullcontext
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
import torchaudio
import numpy as np
from torch.nn import functional as F

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import fade_in_out

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

class TASTE2Stage1Model:
    """Simplified TASTE2 Stage1 Model for audio reconstruction"""
    
    def __init__(self, llm, flow, hift, fp16=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm, self.flow, self.hift = llm, flow, hift
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
    
    def load(self, llm_model, flow_model, hift_model):
        """Load model weights"""
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
    
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


class TASTE2Stage1:
    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):

        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/taste2_stage1.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.audio_extractor = configs['audio_extractor']
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = TASTE2Stage1Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, torch_dtype=torch_dtype, device=device
        )
        
        return pipe(audio_16k.squeeze().numpy())["text"]

    def reconstruct_stage1(self, audio_16k, asr_model_dir=None, text=None):
        """Main reconstruction pipeline for Stage 1"""
        assert asr_model_dir or text, "Either ASR model or text must be provided"
        
        # Get text from ASR or use provided text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        
        # Extract text and audio tokens
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        audio_token, audio_token_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]])
        
        # Generate speech tokens from text and audio
        s3_tokens = list(self.model.llm.inference(
            text_token=text_token.to(self.device),
            text_token_len=text_token_len,
            audio_feature=audio_token.to(self.device),
            audio_feature_len=audio_token_len,
            sampling=25
        ))
        s3_tokens = torch.tensor(s3_tokens, dtype=torch.long).unsqueeze(0) if s3_tokens else torch.zeros(1, 0, dtype=torch.long)
        
        # Convert tokens back to audio
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        audio_chunks = list(self.model.tts(
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding
        ))
        
        return torch.cat([chunk['tts_speech'] for chunk in audio_chunks], dim=1)
    
            
def main():
    parser = argparse.ArgumentParser(description='TASTE2 Stage 1 Audio Reconstruction')
    parser.add_argument('--model_dir', type=str, required=True, help='TASTE2 model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--test_file', type=str, required=True, help='Audio/Arrow file path')
    parser.add_argument('--asr_model_dir', type=str, default="openai/whisper-large-v3", help='ASR model')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load audio data
    audio_16k, ground_truth_text, audio_basename = load_audio_data(args.test_file, args.output_dir)
    
    # Initialize model and run reconstruction
    print("\nInitializing TASTE2Stage1 model...")
    taste_model = TASTE2Stage1(args.model_dir, fp16=False)
    
    print("\nRunning ASR and reconstruction...")
    try:
        asr_text = taste_model._run_asr(audio_16k, args.asr_model_dir)
        print(f"ASR text: {asr_text}")
        
        reconstructed_audio = taste_model.reconstruct_stage1(
            audio_16k, asr_model_dir=args.asr_model_dir
        )
        
        # Save results
        output_path = os.path.join(args.output_dir, f"{audio_basename}_reconstructed.mp3")
        torchaudio.save(output_path, reconstructed_audio.cpu(), taste_model.sample_rate, backend='soundfile')
        
        text_path = os.path.join(args.output_dir, "text_comparison.txt")
        with open(text_path, 'w') as f:
            f.write(f"Original: {ground_truth_text or 'N/A'}\nASR: {asr_text}\n")
        
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
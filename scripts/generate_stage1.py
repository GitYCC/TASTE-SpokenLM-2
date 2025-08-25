
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.file_utils import logging
import taste_speech.taste2
from taste_speech.taste2.taste_decoder import TasteS3GenerationLM


class TASTE2Stage1Model(CosyVoice2Model):
    pass


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
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = TASTE2Stage1Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        
        # Load and filter LLM checkpoint to remove training metadata
        llm_checkpoint = torch.load('{}/llm.pt'.format(model_dir), map_location='cpu')
        if isinstance(llm_checkpoint, dict) and 'model_state_dict' in llm_checkpoint:
            llm_state_dict = llm_checkpoint['model_state_dict']
        elif isinstance(llm_checkpoint, dict):
            # Filter out training metadata keys
            llm_state_dict = {k: v for k, v in llm_checkpoint.items() 
                             if k not in ['epoch', 'step', 'optimizer_state_dict', 'scheduler_state_dict']}
        else:
            llm_state_dict = llm_checkpoint
        
        # Save filtered checkpoint temporarily
        torch.save(llm_state_dict, '{}/llm_filtered.pt'.format(model_dir))
        
        self.model.load('{}/llm_filtered.pt'.format(model_dir),
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
        del configs

    def _run_asr(self,  audio_16k: torch.Tensor, asr_model_dir: str = "openai/whisper-large-v3"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = asr_model_dir

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        result = pipe(audio_16k.squeeze().numpy())  # Squeeze to remove any extra dimensions and convert to numpy
        return result["text"]

    def reconstruct_stage1(
            self, 
            audio_16k: torch.Tensor,
            asr_model_dir: str = None,
            text: str = None, 
        ):
        
        assert asr_model_dir is not None or text is not None
        
        # 1. audio_16k -> (ASR) -> text -> (text encoder) -> text token
        if text is None:
            asr_text = self._run_asr(audio_16k, asr_model_dir)
        else:
            asr_text = text
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        
        # 2. audio_16k -> (audio encoder) -> audio feature
        audio_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.sample_rate)(audio_16k)
        audio_feature, audio_feature_len = self.frontend._extract_speech_feat(audio_resample) 
        
        # 3. gather text token & audio feature and use inference in taste_decoder.py -> speech (audio, s3) token
        
        s3_token_generator = self.llm.inference(
            text_token=text_token,
            text_token_len=text_token_len,
            audio_feature=audio_feature,
            audio_feature_len=audio_feature_len,
            sampling=25
        )
        
        # Collect all s3 tokens from the generator
        s3_tokens = []
        for token in s3_token_generator:
            s3_tokens.append(token)
        s3_tokens = torch.cat(s3_tokens, dim=1) if s3_tokens else torch.tensor([], dtype=torch.long)
        
        # 4. s3token -> (TTS) -> reconstructed_audio_16k
        # Use the model's token2wav method to convert s3 tokens back to audio
        # Extract speaker embedding for TTS
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        
        reconstructed_audio_16k = self.model.token2wav(
            token=s3_tokens,
            prompt_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_feat=torch.zeros(1, 0, 80),
            embedding=speaker_embedding,
            uuid=str(uuid.uuid1()),
            finalize=True
        )
        
        return reconstructed_audio_16k
    
            
if __name__ == '__main__':
    import pyarrow as pa
    import torchaudio
    import numpy as np
    
    model_dir = "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM-2/training/checkpoints/taste2-stage1-scratch-460k"
    asr_model_dir = "openai/whisper-large-v3"
    
    # Create output directory
    output_dir = "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/reconstruction_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test sample from Emilia dataset using HuggingFace datasets
    from datasets import Dataset
    
    test_file = "/mnt/shared/NTU_TASLM/dienruei/data/emilia-en/data/test/emilia-dataset-train-02207-of-04908-taste.arrow"
    
    print("Loading Emilia test sample...")
    dataset = Dataset.from_file(test_file)
    sample = dataset[0]  # Get first sample
    
    print("Dataset keys:", sample.keys())
    
    # Extract audio from mp3 data
    mp3_data = sample['mp3']
    
    print(f"MP3 data type: {type(mp3_data)}")
    print(f"MP3 data keys: {mp3_data.keys() if isinstance(mp3_data, dict) else 'Not a dict'}")
    
    # Extract audio array and sampling rate from mp3 dict
    audio_array = mp3_data['array']
    sr = mp3_data['sampling_rate']
    
    # Convert to torch tensor and resample if needed
    audio_16k = torch.tensor(audio_array, dtype=torch.float32)  # Single channel audio
    if sr != 16000:
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_16k)
    
    # Extract text from JSON field (already a dict)
    text_json = sample['json']
    ground_truth_text = text_json['text']
    
    print(f"Original text: {ground_truth_text}")
    print(f"Audio shape: {audio_16k.shape}")
    audio_length = audio_16k.shape[-1] if audio_16k.dim() > 0 else len(audio_16k)
    print(f"Audio duration: {audio_length / 16000:.2f} seconds")
    
    # Save original audio
    original_audio_path = os.path.join(output_dir, "original_audio.wav")
    audio_to_save = audio_16k.unsqueeze(0) if audio_16k.dim() == 1 else audio_16k
    torchaudio.save(original_audio_path, audio_to_save, 16000)
    print(f"Saved original audio to: {original_audio_path}")
    
    # Initialize TASTE model
    print("\nInitializing TASTE2Stage1 model...")
    taste_model = TASTE2Stage1(model_dir)
    
    # Get ASR transcription first
    print("\nRunning ASR...")
    asr_text = taste_model._run_asr(audio_16k, asr_model_dir)
    print(f"ASR text: {asr_text}")
    
    # Test reconstruction pipeline
    print("\nRunning reconstruction pipeline...")
    try:
        reconstructed_audio = taste_model.reconstruct_stage1(
            audio_16k=audio_16k,
            asr_model_dir=asr_model_dir,
            text=None  # Let it use ASR
        )
        
        print("Reconstruction successful!")
        print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
        
        # Save reconstructed audio
        reconstructed_audio_path = os.path.join(output_dir, "reconstructed_audio.wav")
        torchaudio.save(reconstructed_audio_path, reconstructed_audio.cpu(), 16000)
        print(f"Saved reconstructed audio to: {reconstructed_audio_path}")
        
        # Save text information
        text_info_path = os.path.join(output_dir, "text_comparison.txt")
        with open(text_info_path, 'w') as f:
            f.write(f"Original text: {ground_truth_text}\n")
            f.write(f"ASR text: {asr_text}\n")
        print(f"Saved text comparison to: {text_info_path}")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
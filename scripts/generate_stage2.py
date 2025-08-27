import os
import argparse
import torch
import torchaudio
from huggingface_hub import snapshot_download
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import logging

def load_arrow_file(test_file=None, output_dir=None):
    # Load test sample from Emilia dataset using HuggingFace datasets
    from datasets import Dataset
    assert os.path.exists(test_file)
    
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
    if audio_16k.dim() > 1:
        audio_16k = audio_16k.mean(dim=0, keepdim=True)
    else:
        audio_16k = audio_16k.unsqueeze(dim=0)
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
    assert output_dir is not None
    original_audio_path = os.path.join(output_dir, "audio_original.mp3")
    audio_to_save = audio_16k.unsqueeze(0) if audio_16k.dim() == 1 else audio_16k
    torchaudio.save(original_audio_path, audio_to_save, 16000)
    print(f"Saved original audio to: {original_audio_path}")
    
    return audio_16k, ground_truth_text

def load_audio_file(test_file=None, output_dir=None):
    assert os.path.exists(test_file)
    audio_basename = os.path.basename(test_file).split('.')[0]
    
    print(f"Loading audio file: {test_file}")
    
    # Load audio file using torchaudio
    audio, sr = torchaudio.load(test_file)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_16k = resampler(audio)
    else:
        audio_16k = audio
    
    print(f"Audio shape: {audio_16k.shape}")
    audio_length = audio_16k.shape[-1]
    print(f"Audio duration: {audio_length / 16000:.2f} seconds")
    
    # Save original audio to output directory if specified
    if output_dir is not None:
        original_audio_path = os.path.join(output_dir, f"{audio_basename}_original.wav")
        torchaudio.save(original_audio_path, audio_16k, 16000)
        print(f"Saved original audio to: {original_audio_path}")
    
    return audio_16k, None


class TASTE2Stage2Model(CosyVoice2Model):
    def __init__(self, llm, flow, hift, slm, fp16=False):
        super().__init__(llm, flow, hift, fp16)
        self.slm = slm
        
    def load_slm(self, slm_model):
        state_dict = self.slm.state_dict(torch.load(slm_model, map_location=self.device))
        self.slm.load_state_dict(state_dict, strict=True)
        self.slm.to(self.device).eval()

class TASTE2Stage2:
    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/taste2_stage2.yaml'.format(model_dir)
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
        
        # Initialize model with SLM
        self.model = TASTE2Stage2Model(configs['llm'], configs['flow'], configs['hift'], configs['slm'], fp16)
        
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
        os.remove('{}/llm_filtered.pt'.format(model_dir))
        
        # Load and filter SLM checkpoint to remove training metadata
        llm_checkpoint = torch.load('{}/slm.pt'.format(model_dir), map_location='cpu')
        if isinstance(llm_checkpoint, dict) and 'model_state_dict' in llm_checkpoint:
            llm_state_dict = llm_checkpoint['model_state_dict']
        elif isinstance(llm_checkpoint, dict):
            # Filter out training metadata keys
            llm_state_dict = {k: v for k, v in llm_checkpoint.items() 
                             if k not in ['epoch', 'step', 'optimizer_state_dict', 'scheduler_state_dict']}
        else:
            llm_state_dict = llm_checkpoint
        
        # Save filtered checkpoint temporarily
        torch.save(llm_state_dict, '{}/slm_filtered.pt'.format(model_dir))
        self.model.load_slm('{}/slm_filtered.pt'.format(model_dir))
        os.remove('{}/slm_filtered.pt'.format(model_dir))
        
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

    def generation_stage2(
        self,
        audio_16k: torch.Tensor,
        asr_model_dir: str = None,
        text: str = None
    ):
        """Generate stage 2 output - to be implemented"""
        assert asr_model_dir is not None or text is not None
        # TODO: Implement generation logic
        # 1. audio_16k -> (ASR) -> text -> (text encoder) -> text token
        if text is None:
            asr_text = self._run_asr(audio_16k, asr_model_dir)
        else:
            asr_text = text
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        
        # 2. audio_16k -> (audio extractor) -> audio feature
        audio_token, audio_token_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]]) 
        
        # 3. text token & audio feature -> SLM -> old & new text token, taste emb
        slm_output_generator = self.model.slm.inference(
            text_token=text_token.to(device=self.device),
            text_token_len=text_token_len,
            audio_feature=audio_token.to(device=self.device),
            audio_feature_len=audio_token_len,
            sampling=25,
        )
        
        new_text_tokens, new_taste_embs = list(), list()
        for new_text_token, new_taste_emb in slm_output_generator:
            new_text_tokens.append(new_text_token)
            new_taste_embs.append(new_taste_emb.squeeze())
        new_text_tokens = torch.tensor([new_text_tokens], dtype=torch.int32).to(self.device)
        new_text_tokens_len = torch.tensor([new_text_tokens.shape[1]], dtype=torch.int32).to(self.device)
        new_taste_embs = torch.stack(new_taste_embs).unsqueeze(0).to(self.device)
        
        # 4. gather old, new text token & new taste emb -> llm -> speech (audio, s3) token
        s3_token_generator = self.model.llm.inference(
            text_token=text_token.to(device=self.device),
            text_token_len=text_token_len,
            audio_feature=audio_token.to(device=self.device),
            audio_feature_len=audio_token_len,
            sampling=25,
        )
        
        s3_new_token_generator = self.model.llm.inference_for_taste_emb(
            new_text_tokens,
            new_text_tokens_len,
            new_taste_embs
        )
        
        # Collect all s3 tokens from the generator
        s3_tokens = []
        # default old+new, can change later
        for token in s3_token_generator:
            s3_tokens.append(token)
        for token in s3_new_token_generator:
            s3_tokens.append(token)
        
        s3_tokens = torch.tensor(s3_tokens, dtype=torch.long) if s3_tokens else torch.tensor([], dtype=torch.long)
        s3_tokens = s3_tokens.unsqueeze(dim=0)
        
        # 5. s3token -> (TTS) -> reconstructed_audio_16k
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        
        reconstructed_audio_generator = self.model.tts(
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding,
            stream=False
        )
        audio_chunks = []
        for chunk in reconstructed_audio_generator:
            audio_chunks.append(chunk['tts_speech'])
            
        reconstructed_audio_16k = torch.cat(audio_chunks, dim=1)
        
        return reconstructed_audio_16k

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TASTE2 Stage 2 Audio Reconstruction')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the TASTE2 model directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for reconstructed audio and results')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test audio file (.wav, .mp3) or Arrow file (.arrow)')
    parser.add_argument('--asr_model_dir', type=str, default="openai/whisper-large-v3",
                        help='ASR model directory (default: openai/whisper-large-v3)')
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    output_dir = args.output_dir
    test_file = args.test_file
    asr_model_dir = args.asr_model_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    audio_16k, ground_truth_text = None, None
    
    audio_basename = 'audio'
    if test_file.endswith('.arrow'):
        audio_16k, ground_truth_text = load_arrow_file(test_file, output_dir)
    else: # Expect an audio file (mp3, wav, etc.)
        audio_basename = os.path.basename(test_file).split('.')[0]
        audio_16k, ground_truth_text = load_audio_file(test_file, output_dir)
        
    # Initialize TASTE model
    print("\nInitializing TASTE2Stage2 model...")
    taste_model = TASTE2Stage2(model_dir, fp16=False, load_jit=False)
    
    # Get ASR transcription first
    print("\nRunning ASR...")
    asr_text = taste_model._run_asr(audio_16k, asr_model_dir)
    print(f"ASR text: {asr_text}")
    
    # Test reconstruction pipeline
    print("\nRunning reconstruction pipeline...")
    try:
        reconstructed_audio = taste_model.generation_stage2(
            audio_16k=audio_16k,
            asr_model_dir=asr_model_dir,
            text=None  # Let it use ASR
        )
        
        print("Reconstruction successful!")
        print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
        
        # Save reconstructed audio
        reconstructed_audio_path = os.path.join(output_dir, f"{audio_basename}_reconstructed.mp3")
        torchaudio.save(reconstructed_audio_path, reconstructed_audio.cpu(), sample_rate=taste_model.sample_rate, backend='soundfile')
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

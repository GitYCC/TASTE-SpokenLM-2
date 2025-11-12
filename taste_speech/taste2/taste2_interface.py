import os
import threading
import uuid
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download

from taste_speech.taste2.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from taste_speech.taste2.cosyvoice.utils.file_utils import logging
from taste_speech.taste2.cosyvoice.cli.model import CosyVoice2Model
from taste_speech.taste2.cosyvoice.utils.common import fade_in_out


class TASTE2Model(CosyVoice2Model):
    """Complete TASTE2 Model with full initialization, checkpoint loading, and inference capabilities"""

    def __init__(self, model_dir, stage=1, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dir = model_dir
        self.fp16 = fp16
        self.stage = stage

        # Download model if needed
        if not os.path.exists(model_dir):
            if stage == 1:
                model_dir = snapshot_download(model_dir)
            elif stage == 'sft':
                model_dir = hf_snapshot_download(model_dir)
            else:
                model_dir = hf_snapshot_download(model_dir)

        # Load appropriate config file
        if stage == 'sft':
            config_file = 'taste2_stagesft.yaml'
        else:
            config_file = f'taste2_stage{stage}.yaml'

        hyper_yaml_path = os.path.join(model_dir, config_file)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f'{hyper_yaml_path} not found!')

        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN'),
                    'taste_tokenizer_backbond_path': os.path.join(model_dir, 'distil-whisper'),
                    'qwen_pretrain_path_for_slm': os.path.join(model_dir, 'qwen2-1_5b'),
                    'qwen_pretrain_path_for_slm_7b': os.path.join(model_dir, 'qwen2-7b'),
                }
            )

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

        # Initialize model components
        self.llm = configs['llm']
        self.flow = configs['flow']
        self.hift = configs['hift']
        self.slm = configs.get('slm') if stage in [2, 'sft'] else None

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
        self.hift_cache_dict = {}

        # Load model weights
        self._load_checkpoints(model_dir, stage)

        # Optional optimizations
        if load_vllm:
            self._load_vllm(os.path.join(model_dir, 'vllm'))
        if load_jit:
            precision = 'fp16' if fp16 else 'fp32'
            self._load_jit(os.path.join(model_dir, f'flow.encoder.{precision}.zip'))
        if load_trt:
            precision = 'fp16' if fp16 else 'fp32'
            self._load_trt(
                os.path.join(model_dir, f'flow.decoder.estimator.{precision}.mygpu.plan'),
                os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx'),
                trt_concurrent,
                fp16
            )

        # Set taste_stage1 reference
        self.taste_stage1 = self.slm.taste_stage1 if stage in [2, 'sft'] else self.llm

        del configs

    def _load_checkpoints(self, model_dir, stage):
        """Load model checkpoint weights"""
        # Load LLM
        self.llm.load_state_dict(torch.load(os.path.join(model_dir, 'llm.pt'), map_location=self.device), strict=True)
        self.llm.to(self.device).eval()

        # Load Flow
        self.flow.load_state_dict(torch.load(os.path.join(model_dir, 'flow.pt'), map_location=self.device), strict=True)
        self.flow.to(self.device).eval()

        # Load HiFT (handle HiFiGAN format)
        hift_path = os.path.join(model_dir, 'hift.pt')
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_path, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        # Load SLM if needed
        if stage in [2, 'sft'] and self.slm is not None:
            slm_path = os.path.join(model_dir, 'slm.pt')
            self.slm.load_state_dict(torch.load(slm_path, map_location=self.device), strict=True)
            self.slm.to(self.device).eval()

    def _load_vllm(self, vllm_path):
        """Load VLLM optimizations"""
        from taste_speech.taste2.cosyvoice.utils.file_utils import export_cosyvoice2_vllm
        export_cosyvoice2_vllm(self.llm, vllm_path, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=vllm_path,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def _load_jit(self, jit_path):
        """Load JIT optimizations"""
        # Implementation for JIT loading
        flow_encoder = torch.jit.load(jit_path, map_location=self.device)
        self.flow.encoder = flow_encoder

    def _load_trt(self, trt_plan_path, trt_onnx_path, trt_concurrent, fp16):
        """Load TensorRT optimizations"""
        from taste_speech.taste2.cosyvoice.utils.file_utils import convert_onnx_to_trt
        from taste_speech.taste2.cosyvoice.utils.common import TrtContextWrapper
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(trt_plan_path) or os.path.getsize(trt_plan_path) == 0:
            convert_onnx_to_trt(trt_plan_path, self.get_trt_kwargs(), trt_onnx_path, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(trt_plan_path, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(trt_plan_path)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        """Get TensorRT configuration parameters"""
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def _run_inference_job(self, source_token, session_id, **kwargs):
        """Unified inference job for both LLM and VC"""
        if source_token.shape[1] == 0:  # LLM job
            with self.llm_context, torch.cuda.amp.autocast(self.fp16 and not hasattr(self.llm, 'vllm')):
                for token in self.llm.inference(**{k: v.to(self.device) if hasattr(v, 'to') else v for k, v in kwargs.items()}, uuid=session_id):
                    self.session_data[session_id]['tokens'].append(token)
        else:  # VC job
            self.session_data[session_id]['tokens'] = source_token.flatten().tolist()
        self.session_data[session_id]['finished'] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        if uuid not in self.hift_cache_dict:
            self.hift_cache_dict[uuid] = None

        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

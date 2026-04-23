
# ---------------------------------------------------------------------------
# torchaudio >=2.9 hard-requires torchcodec, which has ABI issues with
# torch 2.9.1+cu130.  Fall back to soundfile-based load/save so every
# downstream call to torchaudio.load / torchaudio.save keeps working.
# ---------------------------------------------------------------------------
def _patch_torchaudio():
    try:
        import torchaudio
        torchaudio.load("__probe__")
    except ImportError:
        import io, os
        from typing import BinaryIO, Optional, Tuple, Union
        import torch, soundfile as sf, numpy as np, torchaudio

        def _sf_load(
            uri: Union[BinaryIO, str, os.PathLike],
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,
            channels_first: bool = True,
            format: Optional[str] = None,
            buffer_size: int = 4096,
            backend: Optional[str] = None,
        ) -> Tuple[torch.Tensor, int]:
            if isinstance(uri, (str, os.PathLike)):
                data, sr = sf.read(str(uri), dtype="float32",
                                   start=frame_offset,
                                   stop=None if num_frames == -1 else frame_offset + num_frames,
                                   always_2d=True)
            else:
                data, sr = sf.read(uri, dtype="float32", always_2d=True)
                if frame_offset > 0 or num_frames != -1:
                    end = None if num_frames == -1 else frame_offset + num_frames
                    data = data[frame_offset:end]
            tensor = torch.from_numpy(data)
            if channels_first:
                tensor = tensor.t()
            return tensor, sr

        def _sf_save(
            uri: Union[BinaryIO, str, os.PathLike],
            src: torch.Tensor,
            sample_rate: int,
            channels_first: bool = True,
            format: Optional[str] = None,
            encoding: Optional[str] = None,
            bits_per_sample: Optional[int] = None,
            buffer_size: int = 4096,
            backend: Optional[str] = None,
            compression: Optional[float] = None,
        ):
            if channels_first:
                src = src.t()
            data = src.cpu().numpy()
            subtype = None
            if bits_per_sample == 16:
                subtype = "PCM_16"
            elif bits_per_sample == 24:
                subtype = "PCM_24"
            elif bits_per_sample == 32:
                subtype = "PCM_32"
            sf.write(str(uri) if isinstance(uri, os.PathLike) else uri,
                     data, sample_rate, subtype=subtype,
                     format=format.upper() if format else None)

        torchaudio.load = _sf_load
        torchaudio.save = _sf_save
    except FileNotFoundError:
        pass
    except Exception:
        pass

_patch_torchaudio()

from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from .configuration_taste import (
    TasteAudioTowerConfig,
    TasteSpeechDecoderConfig,
    TasteSpokenLMConfig,
    TasteConfig,
)
from .modeling_taste import (
    TasteAudioTower,
    TasteSpeechDecoder,
    TasteSpokenLM,
    TasteForCausalLM,
)
from .processing_taste import (
    TasteProcessor
)
from .modules_taste.inference_audio import VoiceGenerator

AutoConfig.register('taste', TasteConfig)
AutoModelForCausalLM.register(TasteConfig, TasteForCausalLM)
AutoProcessor.register(TasteConfig, TasteProcessor)

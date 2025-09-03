import torchaudio
from omegaconf import DictConfig
from torch import nn
import torch

from taste_speech.modules_taste.cosyvoice.flow.flow import MaskedDiffWithXvec
from taste_speech.modules_taste.cosyvoice.hifigan.generator import HiFTGenerator
from taste_speech.modules_taste.cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
from taste_speech.modules_taste.cosyvoice.flow.length_regulator import InterpolateRegulator
from taste_speech.modules_taste.cosyvoice.flow.flow_matching import ConditionalCFM
from taste_speech.modules_taste.cosyvoice.flow.decoder import ConditionalDecoder
from taste_speech.modules_taste.cosyvoice.encoder import ConformerEncoder as CosyVoiceConformerEncoder


class VoiceGenerator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.output_sampling_rate = 22050

        encoder = CosyVoiceConformerEncoder(
            attention_dropout_rate = 0.1,
            input_layer = 'linear',
            pos_enc_layer_type = 'rel_pos_espnet',
            input_size = 512,
            output_size = 512,
            attention_heads = 8,
            use_cnn_module = False,
            macaron_style = False
        )
        length_regulator = InterpolateRegulator(
            channels = 80,
            sampling_ratios = [1, 1, 1, 1]
        )
        decoder = ConditionalCFM(
            in_channels=240,
            spk_emb_dim=8,
            cfm_params=DictConfig({
                    'sigma_min' : 1e-06,
                    'solver' : 'euler',
                    't_scheduler' : 'cosine',
                    'training_cfg_rate' : 0.2,
                    'inference_cfg_rate' : 0.7,
                    'reg_loss_type' : 'l1'
            }),
            estimator = ConditionalDecoder(
                in_channels = 320,
                out_channels = 80,
                dropout = 0,
                n_blocks = 4,
                num_mid_blocks = 12,
                num_heads = 8,
                act_fn = 'gelu'
            )
        )
        self.flow = MaskedDiffWithXvec(
            input_size=512,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=4096,
            input_frame_rate=50,
            only_mask_loss= True,
            encoder=encoder,
            length_regulator=length_regulator,
            decoder=decoder,
            decoder_conf= {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1, 'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}), 'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64, 'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
            mel_feat_conf= {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 16000, 'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}
        )
        self.hift = HiFTGenerator(
            sampling_rate=self.output_sampling_rate,
            f0_predictor=ConvRNNF0Predictor()
        )

    def load_from_cosyvoice_ckpt(self, flow_pt_path, hift_pt_path):
        flow_data = torch.load(flow_pt_path)
        for name, params in self.flow.named_parameters():
            if name in flow_data:
                params.data.copy_(flow_data[name].data)
            else:
                print(f'{name} is missing')

        hift_data = torch.load(hift_pt_path)
        for name, params in self.hift.named_parameters():
            if name in hift_data:
                params.data.copy_(hift_data[name].data)
            else:
                print(f'{name} is missing')

    def inference(self, speech_token_ids, speech_token_lengths, flow_embedding, output_fpath=None):
        device = speech_token_ids.device
        tts_mel = self.flow.inference(
            token=speech_token_ids,
            token_len=speech_token_lengths,
            embedding=flow_embedding.to(device),

            prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device),
            prompt_token_len=torch.zeros(1, dtype=torch.int32).to(device),
            prompt_feat=torch.zeros(1, 0, 80).to(device),
            prompt_feat_len=torch.zeros(1, dtype=torch.int32).to(device),
        )
        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        if output_fpath:
            torchaudio.save(output_fpath, tts_speech, self.output_sampling_rate)
        return tts_speech, self.output_sampling_rate

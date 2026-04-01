
from typing import Dict, Optional, Callable, List, Generator, Tuple
import logging
import time

import torch
from torch import nn
from taste_speech.taste2.cosyvoice.llm.llm import Qwen2LM, th_accuracy, IGNORE_ID


class TasteS3GenerationLM(Qwen2LM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: torch.nn.Module,
        sampling: Callable,
        # = new =
        taste_tokenizer: torch.nn.Module,
        taste_decoder_mixer: torch.nn.Module,
        # =======
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
        
    ):
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
            mix_ratio=mix_ratio,
        )
        self.taste_tokenizer = taste_tokenizer
        self.taste_decoder_mixer = taste_decoder_mixer
        self.weight_commit_loss = 1.0
        self.is_text_only = (taste_tokenizer is None) or (taste_decoder_mixer is None)

    def load_trt_llm(self, trt_llm_path: str, delete_layers: bool = True):
        """Load a TRT-LLM engine and attach it as ``self.trt_llm``.

        This mirrors ``TASTE2Model._load_trt_llm`` but operates directly on
        the ``Qwen2LM`` subclass so ``inference_wrapper`` can dispatch to the
        engine via the ``hasattr(self, 'trt_llm')`` check.

        Args:
            trt_llm_path: Path to the exported HF-style checkpoint directory
                produced by ``export_checkpoint``.
            delete_layers: If *True*, delete the PyTorch transformer layers
                to free GPU memory.  Set to *False* when the PyTorch path
                (e.g. ``inference_bistream``) must remain functional.
        """
        import sys
        import os
        import threading

        accelerate_dir = os.path.join(
            os.path.dirname(__file__), 'cosyvoice', 'trt_llm',
        )
        if accelerate_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(accelerate_dir))

        import cosyvoice_trt_llm  # noqa: F401 — registers CosyVoice2ForCausalLM
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        logging.info("Loading TRT-LLM engine from %s", trt_llm_path)
        self.trt_llm = LLM(
            model=trt_llm_path,
            skip_tokenizer_init=True,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        )

        if not hasattr(self, 'lock'):
            self.lock = threading.Lock()

        if delete_layers:
            del self.llm.model.model.layers
            logging.info("Deleted PyTorch transformer layers to free memory")

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text_token: (B, L)
            text_token_len: (B,)
            audio_feature: 
            audio_feature_len: 
            speech_token: (B, T)
            speech_token_len: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)

        audio_feature = batch['audio_feature'].to(device)
        audio_feature_len = batch['audio_feature_len'].to(device)

        # 1-1. encode text_token
        text_token_emb = self.llm.forward_embed_tokens(text_token).float()

        if not self.is_text_only:
            # 1-2. encode taste_token
            tokenized = self.taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
            taste_token_emb = tokenized['taste_token_emb']

            # 1-3. mixing
            mixed_token_emb = self.taste_decoder_mixer(text_token_emb, taste_token_emb, text_token_len)
        else:
            mixed_token_emb = text_token_emb

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(text_token, mixed_token_emb, text_token_len, speech_token, speech_token_emb, speech_token_len)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output.float())
        loss = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)

        if not self.is_text_only and 'commit_loss' in tokenized:
            loss += self.weight_commit_loss * tokenized['commit_loss']

        return {'loss': loss, 'acc': acc}

    @torch.inference_mode()
    def inference(
        self,
        text_token: torch.Tensor,
        text_token_len: torch.Tensor,
        audio_feature: Optional[torch.Tensor] = None,
        audio_feature_len: Optional[torch.Tensor] = None,
        taste_token_emb: Optional[torch.Tensor] = None,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = '',
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:

        assert (taste_token_emb is not None) ^  (audio_feature is not None and audio_feature_len is not None)

        text_token_emb = self.llm.forward_embed_tokens(text_token).float()

        if not self.is_text_only:
            # 1-2. encode taste_token
            if taste_token_emb is None:
                tokenized = self.taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
                taste_token_emb = tokenized['taste_token_emb']

            # 1-3. mixing
            mixed_token_emb = self.taste_decoder_mixer(text_token_emb, taste_token_emb, text_token_len)
        else:
            mixed_token_emb = text_token_emb

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos_eos_emb, mixed_token_emb, task_id_emb], dim=1)

        # 4. cal min/max_length
        min_len = int(text_token_len * min_token_text_ratio)
        max_len = int(text_token_len * max_token_text_ratio)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_bistream(
        self,
        input_generator: Generator[Tuple[torch.Tensor, torch.Tensor], None, None], # text_token_ids (Shape: [1, seq_len]), taste_embs (Shape: [1, seq_len, emb_dim]])
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_feature: torch.Tensor,
        prompt_speech_feature_len: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = '',
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]: # s3_token_ids (size=[1, s3_len])
        
        device = prompt_text.device
        
        # 1. Prepare initial input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos_eos_emb], dim=1)
        
        # 2. Process prompt to get initial mixed embeddings (treat as part of text_cache)
        if prompt_text.size(1) > 0:
            prompt_text_emb = self.llm.forward_embed_tokens(prompt_text).float()
            
            if not self.is_text_only and prompt_speech_feature_len > 0:
                # Use taste_tokenizer to generate taste embedding from prompt audio features
                tokenized = self.taste_tokenizer(prompt_text, prompt_text_len, prompt_speech_feature, prompt_speech_feature_len)
                prompt_taste_emb = tokenized['taste_token_emb']
                
                # Mix prompt text and taste embeddings
                text_cache = self.taste_decoder_mixer(prompt_text_emb, prompt_taste_emb, prompt_text_len)
            else:
                text_cache = prompt_text_emb
        else:
            text_cache = torch.zeros(1, 0, self.llm_input_size, dtype=torch.float32).to(device)
        
        # 3. Bistream generation
        if hasattr(self, 'trt_llm'):
            # TRT-LLM path: rebuild full embedding history per segment,
            # call trt_llm.generate() with fill_token as stop for each segment.
            from tensorrt_llm.sampling_params import SamplingParams as TrtSamplingParams

            fill_token_id = self.speech_token_size + 2
            eos_token_id = self.speech_token_size
            history_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=torch.float32,
            ).to(device)
            out_tokens = []

            for text_token, taste_emb in input_generator:
                logging.debug(f"[TRT-BISTREAM] received text_token shape={text_token.shape}")

                text_emb = self.llm.forward_embed_tokens(text_token).float()
                text_len = torch.tensor([text_token.size(1)], device=device)

                if not self.is_text_only and taste_emb is not None:
                    mixed_emb = self.taste_decoder_mixer(text_emb, taste_emb, text_len)
                else:
                    mixed_emb = text_emb

                text_cache = torch.cat([text_cache, mixed_emb], dim=1)

                if text_cache.size(1) < self.mix_ratio[0]:
                    continue

                # Consume mix_ratio[0] text embeddings and append to history
                lm_input_text = text_cache[:, :self.mix_ratio[0]]
                text_cache = text_cache[:, self.mix_ratio[0]:]
                history_emb = torch.cat([history_emb, lm_input_text], dim=1)

                full_input = torch.cat([sos_eos_emb, history_emb], dim=1)
                logging.info(
                    '[TRT-BISTREAM] segment generate, history_len=%d, total_out=%d',
                    history_emb.size(1), len(out_tokens),
                )

                sampling_params = TrtSamplingParams(
                    max_tokens=self.mix_ratio[1] + 1,
                    end_id=fill_token_id,
                    top_k=sampling,
                )
                inputs = {
                    "prompt": "",
                    "multi_modal_embeddings": {
                        "speech": [full_input.squeeze(0).to(torch.bfloat16)],
                    },
                }
                output = self.trt_llm.generate(inputs, sampling_params=sampling_params)

                speech_embs = []
                for token_id in output.outputs[0].token_ids:
                    if token_id == fill_token_id:
                        out_tokens.append(token_id)
                        break
                    if token_id >= self.speech_token_size:
                        break
                    out_tokens.append(token_id)
                    yield token_id
                    speech_embs.append(
                        self.speech_embedding.weight[token_id].reshape(1, 1, -1)
                    )

                if speech_embs:
                    history_emb = torch.cat([history_emb] + speech_embs, dim=1)
                logging.info(
                    '[TRT-BISTREAM] segment done, generated %d speech tokens',
                    len(speech_embs),
                )

            # Final decode: append remaining text + task_id, generate until EOS
            history_emb = torch.cat([history_emb, text_cache, task_id_emb], dim=1)
            full_input = torch.cat([sos_eos_emb, history_emb], dim=1)
            logging.info(
                '[TRT-BISTREAM] final decode, total_input_len=%d',
                full_input.size(1),
            )

            sampling_params = TrtSamplingParams(
                max_tokens=2000,
                end_id=eos_token_id,
                top_k=sampling,
            )
            inputs = {
                "prompt": "",
                "multi_modal_embeddings": {
                    "speech": [full_input.squeeze(0).to(torch.bfloat16)],
                },
            }
            output = self.trt_llm.generate(inputs, sampling_params=sampling_params)

            for token_id in output.outputs[0].token_ids:
                if token_id >= self.speech_token_size:
                    if token_id == eos_token_id:
                        logging.info(
                            '[TRT-BISTREAM] EOS, total_tokens=%d',
                            len(out_tokens),
                        )
                    break
                out_tokens.append(token_id)
                yield token_id

        else:
            # PyTorch forward_one_step path (no TRT-LLM)
            out_tokens = []
            cache = None
            next_fill_index = -1

            for text_token, taste_emb in input_generator:
                logging.debug(f"[CHECKPOINT:S3-INPUT-RECEIVED] text_token_shape={text_token.shape}")

                text_emb = self.llm.forward_embed_tokens(text_token).float()
                text_len = torch.tensor([text_token.size(1)], device=device)
                logging.debug(f"[CHECKPOINT:S3-TEXT-EMB-DONE] text_emb_shape={text_emb.shape}")

                if not self.is_text_only and taste_emb is not None:
                    mixed_emb = self.taste_decoder_mixer(text_emb, taste_emb, text_len)
                    logging.debug(f"[CHECKPOINT:S3-MIXER-DONE] mixed_emb_shape={mixed_emb.shape}")
                else:
                    mixed_emb = text_emb

                text_cache = torch.concat([text_cache, mixed_emb], dim=1)
                logging.debug(f"[CHECKPOINT:S3-CACHE-UPDATE] text_cache_size={text_cache.size(1)}")

                if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('[CHECKPOINT:S3-FILL-TOKEN] get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('[CHECKPOINT:S3-APPEND-TEXT] append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('[CHECKPOINT:S3-WAIT] not enough text token to decode, wait for more')
                        continue

                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)

                    logging.debug(f"[CHECKPOINT:S3-LLM-FORWARD-START] seq_len={seq_len}")
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                              masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                              cache=cache)
                    logging.debug(f"[CHECKPOINT:S3-LLM-FORWARD-END] y_pred_shape={y_pred.shape}")

                    logp = self.llm_decoder(y_pred[:, -1].float()).log_softmax(dim=-1)

                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.speech_token_size + 2
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True).item()

                    if top_ids == self.speech_token_size + 2:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('[CHECKPOINT:S3-FILL-TOKEN-GEN] fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))

                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.speech_token_size + 2:
                            logging.debug(f"[CHECKPOINT:S3-BREAK] Breaking generation loop")
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    logging.debug(f"[CHECKPOINT:S3-YIELD-TOKEN] token_id={top_ids}, total_tokens={len(out_tokens)}")
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

            # Final decode
            lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
            logging.info('[CHECKPOINT:S3-FINAL-DECODE] no more text token, decode until met eos')
            while True:
                seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)

                logging.debug(f"[CHECKPOINT:S3-FINAL-LLM-START] seq_len={seq_len}")
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                          masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                          cache=cache)
                logging.debug(f"[CHECKPOINT:S3-FINAL-LLM-END] y_pred_shape={y_pred.shape}")

                logp = self.llm_decoder(y_pred[:, -1].float()).log_softmax(dim=-1)
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False).item()
                out_tokens.append(top_ids)
                if top_ids >= self.speech_token_size:
                    if top_ids == self.speech_token_size:
                        logging.info(f"[CHECKPOINT:S3-EOS] Reached EOS, total_tokens={len(out_tokens)}")
                        break
                    else:
                        raise ValueError('should not get token {}'.format(top_ids))
                logging.debug(f"[CHECKPOINT:S3-FINAL-YIELD] token_id={top_ids}, total_tokens={len(out_tokens)}")
                yield top_ids
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

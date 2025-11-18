
"""TASTE Speech Language Model (SLM) Module

This module implements the TASTE (Text-Audio Speech Token Enhancement) Speech Language Model,
which is designed for multimodal speech synthesis and understanding. The model combines text
token embeddings with "taste" token embeddings (derived from audio features) to generate
both text predictions and speech representations.

The module consists of three main components:
1. TasteSLMFusing: Fuses text and taste token embeddings with temporal alignment
2. TasteSLMOut: Handles dual-output prediction for text and taste modalities
3. TasteSLM: Main model orchestrating the entire pipeline

The architecture supports variational latent representation learning for speech tokens
and maintains temporal coherence through a delay mechanism.
"""

import logging

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Callable, Generator
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from taste_speech.taste2.cosyvoice.utils.common import th_accuracy
from taste_speech.modules_taste.fusion import TTS_INPUT_FUSION_CLASSES
from taste_speech.taste2.taste_sft_utils import expand_conversations_to_messages, reconstruct_conversations_from_messages, extract_taste_from_message

# Constants
IGNORE_ID = -1


class TasteSLMFusing(nn.Module):
    """Fusion module for combining text and taste token embeddings.
    
    This module handles the temporal alignment and fusion of text token embeddings
    and taste token embeddings. It introduces a configurable delay mechanism to
    align the modalities and uses learnable padding embeddings for both text and
    taste tokens when needed.
    """
    def __init__(
        self,
        llm_input_size: int,
        tokenizer_output_size: int = None,
        class_name: str = 'weighted_sum',
        fuse_config = dict(),
    ):
        """Initialize the TasteSLMFusing module.
        
        Args:
            llm_input_size (int): Dimension size of the language model input embeddings
            class_name (str): Name of the fusion class to use from TTS_INPUT_FUSION_CLASSES.
                             Default is 'weighted_sum'
        """
        super().__init__()
        if tokenizer_output_size is None:
            tokenizer_output_size = llm_input_size

        # Initialize the fusion mixer based on the specified class name
        self.mixer = TTS_INPUT_FUSION_CLASSES[class_name](**fuse_config)
        
        # Learnable padding embeddings for taste tokens when delay padding is needed
        self.pad_taste_embed = nn.parameter.Parameter(
            torch.zeros(llm_input_size, dtype=torch.float32)
        )
        
        # Learnable padding embeddings for text tokens when delay padding is needed
        self.pad_text_embed = nn.parameter.Parameter(
            torch.zeros(llm_input_size, dtype=torch.float32)
        )

        self.taste_embed_in = nn.Linear(tokenizer_output_size, llm_input_size, bias=True)

    def forward(
        self,
        text_token_emb: torch.Tensor,  # Shape: (1, text_len, embed_dim)
        taste_token_emb: torch.Tensor,  # Shape: (1, taste_len, embed_dim)
        text_token_len: torch.Tensor,  # Shape: (1,) - length of text tokens
        delay: int,  # Number of time steps to delay taste tokens
    ) -> torch.Tensor:
        """Forward pass for fusing text and taste token embeddings.
        
        This method performs temporal alignment by shifting taste tokens forward
        in time (adding delay padding at the beginning) and text tokens backward
        in time (adding delay padding at the end). This alignment ensures proper
        synchronization between modalities.
        
        Args:
            text_token_emb: Text token embeddings with batch size 1
            taste_token_emb: Taste token embeddings with batch size 1
            text_token_len: Length of valid text tokens
            delay: Number of time steps to delay the taste tokens
            
        Returns:
            torch.Tensor: Fused embeddings combining both modalities
        """
        # Ensure batch size is 1 as expected
        assert text_token_emb.size(0) == 1
        
        # Shift taste tokens forward in time by adding delay padding at the beginning
        # This aligns taste tokens with the corresponding text tokens in the future
        shifted_taste_token_emb = torch.concat(
            [
                self.pad_taste_embed.unsqueeze(0).unsqueeze(0).repeat(1, delay, 1),
                self.taste_embed_in(taste_token_emb),
            ], dim=1
        )
        
        # Shift text tokens backward in time by adding delay padding at the end
        # This ensures the sequence lengths match after shifting
        shifted_text_token_emb = torch.concat(
            [
                text_token_emb,
                self.pad_text_embed.unsqueeze(0).unsqueeze(0).repeat(1, delay, 1),
            ], dim=1
        )
        
        # Fuse the aligned embeddings using the configured mixer
        # Returns the first output from the mixer (fused embeddings)
        fused_emb, _ = self.mixer(
            shifted_taste_token_emb,
            text_token_len + delay,  # Adjusted length after shifting
            shifted_text_token_emb,
            text_token_len + delay,  # Adjusted length after shifting
        )
        return fused_emb


class TasteSLMOut(nn.Module):
    """Output module for dual-modal prediction in TASTE SLM.
    
    This module handles the output layer of the TASTE Speech Language Model, providing
    both text token prediction (via cross-entropy loss) and taste latent representation
    learning (via MSE loss and KL divergence). It implements a variational approach
    for taste token generation using reparameterization trick.
    
    The module combines:
    - Text vocabulary prediction with standard language modeling loss
    - Variational latent space learning for taste (speech) tokens
    - KL divergence regularization for proper latent distribution learning
    """
    def __init__(
        self,
        llm_output_size: int,
        text_vocab_size: int,
        d: int,
        fc_mu_requires_bias: bool = True,
        b_logvar_is_linear: bool = False,
        conduct_reparameterization: bool = True,
    ):
        """Initialize the TasteSLMOut module.
        
        Args:
            llm_output_size (int): Dimension of the language model output features
            text_vocab_size (int): Size of the text vocabulary for classification
            d (int): Dimension of the taste latent space
            taste_tokenizer (torch.nn.Module): Tokenizer module for taste tokens
            fc_mu_requires_bias (bool): Whether to use bias in the mu projection layer
            b_logvar_is_linear (bool): Whether to use linear layer for logvar (vs parameter)
            conduct_reparameterization (bool): Whether to apply reparameterization during training
        """
        super().__init__()

        # Store configuration parameters
        self.text_vocab_size = text_vocab_size
        self.b_logvar_is_linear = b_logvar_is_linear
        self.fc_mu_requires_bias = fc_mu_requires_bias
        self.conduct_reparameterization = conduct_reparameterization

        # Loss modules for text and taste predictions
        self.ce_loss_module = nn.CrossEntropyLoss(reduction="mean", ignore_index=IGNORE_ID)
        self.mse_loss_module = nn.MSELoss()

        # Variational layers for taste latent space
        # Linear layer to predict mean (mu) of the latent distribution
        self.fc_mu = nn.Linear(llm_output_size, d, bias=fc_mu_requires_bias)
        
        # Log variance can be either learnable per-sample or global parameter
        if b_logvar_is_linear:
            # Per-sample log variance prediction
            self.b_logvar = nn.Linear(llm_output_size, d, bias=False)
        else:
            # Global learnable log variance parameter
            self.b_logvar = nn.Parameter(torch.zeros(d))

    def _calculate_loss_text_ce(self, text_logits: torch.Tensor, text_labels: torch.Tensor) -> torch.Tensor:
        """Calculate cross-entropy loss for text token prediction.
        
        Args:
            text_logits (torch.Tensor): Predicted logits for text tokens, shape (B, T, vocab_size)
            text_labels (torch.Tensor): Ground truth text token labels, shape (B, T)
            
        Returns:
            torch.Tensor: Cross-entropy loss value
        """
        # Reshape logits and labels for cross-entropy computation
        B, T, C = text_logits.shape
        # Flatten batch and time dimensions for standard cross-entropy loss
        ce_loss = self.ce_loss_module(text_logits.view((B * T, C)), text_labels.view((B * T,)))
        return ce_loss

    def _calculate_loss_taste_mse(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        lm_taste_latent_target: torch.Tensor,
        lm_taste_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate MSE and KL divergence losses for taste latent prediction.
        
        This method computes the taste loss by:
        1. Reconstructing target embeddings from the VQ module
        2. Computing MSE loss between predicted and target latents
        3. Computing KL divergence loss for regularization
        
        Args:
            z (torch.Tensor): Predicted latent representations
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            lm_taste_latent_target (torch.Tensor): Target taste latents
            lm_taste_mask (torch.Tensor): Mask indicating valid taste positions
            
        Returns:
            torch.Tensor: Combined taste loss (MSE + KL divergence)
        """

        # MSE reconstruction loss between predicted latents and targets
        l_reg = self.mse_loss_module(z[lm_taste_mask], lm_taste_latent_target[lm_taste_mask])
        
        # KL divergence loss for regularization
        # Apply mask to logvar to match the shape of other masked tensors
        # Note: logvar is already expanded to (B, T, d) shape in predict_taste_latent
        logvar_masked = logvar[lm_taste_mask]
        
        # KL divergence: 
        l_kl = 0.5 * torch.mean(
            torch.exp(logvar_masked) + 
            (mu[lm_taste_mask] - lm_taste_latent_target[lm_taste_mask])**2 * torch.exp(-logvar_masked) - 
            1 - 
            logvar_masked
        )

        # Combine MSE and KL losses with equal weighting
        taste_loss = 0.5 * l_reg + 0.5 * l_kl
        return taste_loss

    def _reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for variational sampling.
        
        This method implements the standard reparameterization trick used in
        Variational Autoencoders (VAE): z = μ + σ * ε, where ε ~ N(0,1).
        This allows gradients to flow through the sampling process during training.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            sigma (torch.Tensor): Standard deviation of the latent distribution
            
        Returns:
            torch.Tensor: Reparameterized sample z = μ + σ * ε
        """
        # Sample random noise from standard normal distribution
        epsilon = torch.randn_like(sigma)
        # Apply reparameterization trick: z = μ + σ * ε
        z = mu + sigma * epsilon
        return z

    def predict_taste_latent(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict taste latent representations from hidden states.
        
        This method computes the mean and variance of the latent distribution
        and applies reparameterization during training for stochastic sampling.
        
        Args:
            hidden (torch.Tensor): Hidden states from the language model
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (z, mu, logvar)
                - z: Predicted taste latent representations
                - mu: Mean of the latent distribution
                - logvar: Log variance of the latent distribution
        """
        # Compute the mean of the latent distribution
        mu = self.fc_mu(hidden)

        # Compute log variance and convert to standard deviation
        if self.b_logvar_is_linear:
            # Per-sample log variance prediction
            logvar = self.b_logvar(hidden)
            sigma = torch.exp(0.5 * logvar)
        else:
            # Global log variance parameter expanded to match input shape
            batch_size, seq_len, _ = mu.shape
            logvar = self.b_logvar.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            sigma = torch.exp(0.5 * logvar)

        # Apply reparameterization trick during training for stochastic sampling
        if self.conduct_reparameterization:
            z = self._reparameterize(mu, sigma)
        else:
            # Deterministic prediction: use mean + standard deviation
            z = mu
        return z, mu, logvar

    def forward(
        self,
        lm_output: torch.Tensor, 
        text_logit: torch.Tensor, 
        lm_text_target: torch.Tensor, 
        lm_taste_latent_target: torch.Tensor, 
        lm_taste_mask: torch.Tensor,
    ) -> dict:
        """Forward pass computing both text and taste losses.
        
        Args:
            lm_output (torch.Tensor): Output hidden states from language model
            text_logit (torch.Tensor): Predicted text logits
            lm_text_target (torch.Tensor): Target text tokens
            lm_taste_latent_target (torch.Tensor): Target taste latents
            lm_taste_mask (torch.Tensor): Mask for valid taste positions
            
        Returns:
            dict: Dictionary containing loss, accuracy, and prediction values
        """
        # Compute text prediction loss using cross-entropy
        text_loss = self._calculate_loss_text_ce(text_logit, lm_text_target)

        # Predict taste latent representations and get mu, logvar
        z, mu, logvar = self.predict_taste_latent(lm_output)
        # Compute taste prediction loss (MSE + KL divergence)
        taste_loss = self._calculate_loss_taste_mse(z, mu, logvar, lm_taste_latent_target, lm_taste_mask)

        # Combine losses with equal weighting
        loss = 0.5 * text_loss + 0.5 * taste_loss
        # Calculate text prediction accuracy
        text_acc = th_accuracy(text_logit.view(-1, self.text_vocab_size), lm_text_target, ignore_label=IGNORE_ID)
        
        return {
            'loss': loss,
            'text_acc': text_acc,
            'taste_loss': taste_loss,
        }


class TasteSLM(nn.Module):
    """Main TASTE Speech Language Model integrating all components.

    This is the primary model class that orchestrates the entire TASTE pipeline:
    1. Text token embedding through the speech language model
    2. Taste token embedding through the taste tokenizer
    3. Multimodal fusion with temporal alignment
    4. Language model processing
    5. Dual-modal output prediction (text + taste)

    The model handles batch processing with proper sequence padding and supports
    both training and inference modes with configurable delay mechanisms.
    """
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        d: int,
        taste_stage1: torch.nn.Module,
        slm: torch.nn.Module,
        fusing_module: torch.nn.Module,
        out_module: torch.nn.Module,
        path_reload_taste_stage1: str = '',
        delay: int = 1,
        use_continue: bool = False,
        ignore_id: int = -1,
        eos_token_id: int = 151643,
        im_end_token_id: int = 151645,
        text_sampling_callable: Callable = None,
        banned_token_ids: list = None,
    ):
        """Initialize the TASTE Speech Language Model.
        
        Args:
            llm_input_size (int): Input dimension size for the language model
            llm_output_size (int): Output dimension size from the language model
            d (int): Dimension of the taste latent space
            taste_tokenizer (torch.nn.Module): Module for encoding audio to taste tokens
            slm (torch.nn.Module): Speech language model backbone
            fusing_module (torch.nn.Module): Module for fusing text and taste embeddings
            out_module (torch.nn.Module): Output module for dual-modal prediction
            delay (int): Temporal delay for aligning modalities. Default: 1
            ignore_id (int): Token ID to ignore in loss computation. Default: -1
            eos_token_id (int): End-of-sequence token ID. Default: 151643
        """
        super().__init__()
        assert delay > 0

        # Register taste tokenizer
        if path_reload_taste_stage1:
            taste_stage1 = self._reload_taste_stage1(taste_stage1, path_reload_taste_stage1)
        self.taste_stage1 = taste_stage1

        # Core model components
        self.slm = slm  # Speech language model backbone
        self.fusing_module = fusing_module  # Multimodal fusion module
        self.out_module = out_module  # Output prediction module
        if use_continue:
            assert taste_stage1.taste_tokenizer.neck_on is True

        # Configuration parameters
        self.d = d
        self.delay = delay  # Temporal alignment delay
        self.use_continue = use_continue
        self.ignore_id = ignore_id  # Token ID to ignore in loss computation
        self.eos_token_id = eos_token_id  # End-of-sequence token
        self.im_end_token_id = im_end_token_id  # <|im_end|> token ID for masking overlaps

        self.text_sampling_callable = text_sampling_callable

        # Token banning for inference (e.g., ban numbers and non-speech punctuation)
        self.banned_token_ids = banned_token_ids if banned_token_ids is not None else []

    def reload(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint, strict=True)

    def _reload_taste_stage1(self, taste_stage1, path_reload_taste_stage1):
        checkpoint = torch.load(path_reload_taste_stage1, map_location='cpu')
        taste_stage1.load_state_dict(checkpoint, strict=True)
        return taste_stage1

    def prepare_lm_input_target(
        self,
        text_token: torch.Tensor,
        text_token_emb: torch.Tensor,
        text_token_len: torch.Tensor,
        lm_input_target_mode: str,
        taste_latent: torch.Tensor = None,
        taste_token_emb: torch.Tensor = None,
        overlap_mask: Optional[torch.Tensor] = None,
        token_message_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Initialize lists to collect processed sequences
        lm_text_target, lm_taste_latent_target, lm_taste_mask, lm_input = [], [], [], []
        
        # Unpad sequences to remove batch-level padding and process individually
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        taste_latent = unpad_sequence(taste_latent, text_token_len.cpu(), batch_first=True)
        taste_token_emb = unpad_sequence(taste_token_emb, text_token_len.cpu(), batch_first=True)
        
        # Process each sequence in the batch individually
        for i in range(len(text_token)):
            # Create text target: shift tokens by 1, add EOS, pad with ignore tokens for delay
            # Target length should match input length: text_len + delay
            this_lm_text_target = torch.tensor(
                text_token[i].tolist()[1:] + [self.eos_token_id] + [self.ignore_id] * (self.delay - 1)
            )

            # Mask <im_end> tokens for overlapped messages in SFT mode
            if overlap_mask is not None and token_message_ids is not None:
                sample_overlap_mask = overlap_mask[i]  # [M] - which messages are overlapped
                sample_token_msg_ids = token_message_ids[i]  # [L] - which message each token belongs to

                # Find which messages are overlapped (overlap_mask[msg_id] == 1)
                overlapped_msg_ids = torch.where(sample_overlap_mask == 1)[0]

                for msg_id in overlapped_msg_ids:
                    # Find positions where this message's content tokens are
                    msg_content_mask = (sample_token_msg_ids == msg_id)
                    msg_positions = msg_content_mask.nonzero(as_tuple=True)[0]

                    if len(msg_positions) > 0:
                        # The <im_end> token is right after the last content token of this message
                        # It has token_message_ids == -1 (special token)
                        last_content_pos = msg_positions[-1].item()

                        # Search for <im_end> token in the next few positions
                        for check_pos in range(last_content_pos + 1, min(last_content_pos + 5, len(text_token[i]))):
                            if text_token[i][check_pos].item() == self.im_end_token_id:
                                # Found <im_end> token, mask it in target
                                # Remember: target is shifted by -1 (target = input[1:])
                                target_pos = check_pos - 1
                                if 0 <= target_pos < len(this_lm_text_target):
                                    this_lm_text_target[target_pos] = self.ignore_id
                                break

            this_taste_latent_target = torch.tensor(
                [[0.0 for _ in range(self.d)] for _ in range(self.delay - 1)] + taste_latent[i].tolist()
            )
            
            # Create mask for taste tokens: 0 for delay positions, 1 for valid taste positions
            # Mask length should match input length: taste_len + delay
            if lm_input_target_mode == "sft":
                # For SFT mode: mask out zero embeddings (positions without real audio data)
                has_real_data = torch.any(taste_token_emb[i] != 0, dim=-1)  # [seq_len]
                this_lm_taste_mask = torch.tensor(
                    [False] * (self.delay - 1) + has_real_data.tolist()
                )
            else: 
                this_lm_taste_mask = torch.tensor(
                    [False] * (self.delay - 1) + [True] * taste_token_emb[i].size(0)
                )
            
            # Fuse text and taste embeddings for this sequence
            this_lm_input = self.fusing_module(text_token_emb[i].unsqueeze(0), taste_token_emb[i].unsqueeze(0), text_token_len[i:i+1], self.delay)[0, :-1]

            # Collect all processed sequences (remove the batch dimension for pad_sequence)
            lm_text_target.append(this_lm_text_target)
            lm_taste_latent_target.append(this_taste_latent_target)
            lm_taste_mask.append(this_lm_taste_mask)
            lm_input.append(this_lm_input)
        
        # Calculate lengths of fused sequences and re-pad for batch processing
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=0.0)
        lm_text_target = pad_sequence(lm_text_target, batch_first=True, padding_value=self.ignore_id)
        lm_taste_latent_target = pad_sequence(lm_taste_latent_target, batch_first=True, padding_value=0.0)
        lm_taste_mask = pad_sequence(lm_taste_mask, batch_first=True, padding_value=False)
        
        return lm_text_target, lm_taste_latent_target, lm_taste_mask, lm_input, lm_input_len

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> dict:
        """Forward pass of the TASTE Speech Language Model.
        
        This method orchestrates the complete pipeline:
        1. Extract and move input data to the specified device
        2. Encode text tokens using the SLM's embedding layer
        3. Encode audio features to taste token embeddings
        4. Prepare aligned inputs and targets for language model training
        5. Run language model forward pass
        6. Compute dual-modal predictions and losses
        
        Args:
            batch (dict): Input batch containing:
                - text_token: Text token IDs, shape (B, L)
                - text_token_len: Text sequence lengths, shape (B,)
                - speech_token: Speech token IDs, shape (B, T)  
                - speech_token_len: Speech sequence lengths, shape (B,)
                - audio_feature: Raw audio features for taste tokenization
                - audio_feature_len: Audio feature lengths, shape (B,)
            device (torch.device): Target device for computation
            
        Returns:
            dict: Dictionary containing:
                - loss: Combined training loss
                - text_acc: Text prediction accuracy
                - taste_loss: Taste-specific loss component
                - z: Predicted taste latent representations
        """
        if batch.get('sft_training') is True:
            # Switch lm_target to sft
            lm_input_target_mode = "sft"
            
            # Data for recontruct
            full_text_token = batch['text_token']  # [B, L]
            full_text_token_len = batch['text_token_len']  # [B]
            full_token_message_ids = batch['token_message_ids']  # [B, L]
            
            # Prepare data for taste_stage1 (expand dialogue batch to message batch)
            text_token, text_token_len, audio_feature, audio_feature_len, message_mapping = expand_conversations_to_messages(batch, device)
        else:
            # Switch lm_target to pretrain
            lm_input_target_mode = "pretrain"
            # Extract and move input tensors to the specified device
            text_token = batch['text_token'].to(device)
            text_token_len = batch['text_token_len'].to(device)
            audio_feature = batch['audio_feature'].to(device)
            audio_feature_len = batch['audio_feature_len'].to(device)

        # Encode audio features to taste token embeddings using taste tokenizer
        tokenized = self.taste_stage1.taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']
        taste_latent = tokenized['taste_latent']
        

        if batch.get('sft_training') is True:
            # Recontruct for slm training (concat message batch to dialogue batch)
            text_token, text_token_len, taste_token_emb, taste_latent = \
            reconstruct_conversations_from_messages(taste_token_emb, taste_latent, message_mapping,
                            full_text_token, full_text_token_len, full_token_message_ids,
                            text_token_len, device)

        text_token_emb = self.slm.forward_embed_tokens(text_token)

        # Extract overlap_mask and token_message_ids from batch for SFT mode
        overlap_mask = batch.get('overlap_mask', None) if batch.get('sft_training') else None
        token_message_ids = batch.get('token_message_ids', None) if batch.get('sft_training') else None

        # Prepare aligned language model inputs and targets
        lm_text_target, lm_taste_latent_target, lm_taste_mask, lm_input, lm_input_len = \
            self.prepare_lm_input_target(text_token, text_token_emb.float(), text_token_len, lm_input_target_mode,
                taste_latent=taste_latent, taste_token_emb=taste_token_emb,
                overlap_mask=overlap_mask, token_message_ids=token_message_ids)
        
        # Move prepared tensors to device
        lm_text_target = lm_text_target.to(device)
        lm_taste_latent_target = lm_taste_latent_target.to(device)
        lm_taste_mask = lm_taste_mask.to(device)
        lm_input_len = lm_input_len.to(device)

        # Step 4: Run language model forward pass
        lm_output, lm_output_mask = self.slm(lm_input, lm_input_len)
        # Generate text logits using the language model head
        text_logit = self.slm.forward_lm_head(lm_output)
        
        # Step 5: Compute dual-modal outputs and losses
        outputs = self.out_module(
            lm_output.float(), text_logit, 
            lm_text_target, lm_taste_latent_target, lm_taste_mask, 
        )
        # Output dictionary includes: loss, text_acc, taste_loss, z
        return outputs

    @torch.inference_mode()
    def inference(
        self,
        text_token: torch.Tensor,
        text_token_len: torch.Tensor,
        audio_feature: Optional[torch.Tensor] = None,
        audio_feature_len: Optional[torch.Tensor] = None,
        taste_token_emb: Optional[torch.Tensor] = None,
        formatted_text_token: Optional[torch.Tensor] = None,        #only for sft mode
        formatted_text_token_len: Optional[torch.Tensor] = None,    #only for sft mode
        token_message_ids:Optional[torch.Tensor] = None,    #only for sft mode
        sampling: int = 25,
        max_len: int = 20,
        min_len: int = 5,
        uuid: str = '',
        active_rule: Callable = None,
        stop_id: int = None,
        **kwargs,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        assert text_token.size(0) == 1
        assert (taste_token_emb is not None) ^  (audio_feature is not None and audio_feature_len is not None)

        text_token_emb = self.slm.forward_embed_tokens(text_token).float()

        # 1-2. encode taste_token
        if taste_token_emb is None:
            tokenized = self.taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
            taste_token_emb = tokenized['taste_token_emb']

        # Reconstruct taste embeddings for SFT mode if needed
        if formatted_text_token is not None and token_message_ids is not None:
            text_token_emb, taste_token_emb, text_token, text_token_len = extract_taste_from_message(
                self.slm, taste_token_emb, formatted_text_token, formatted_text_token_len, token_message_ids
            )

        # lm_input
        fused = self.fusing_module(text_token_emb, taste_token_emb, text_token_len, self.delay)
        lm_input = fused[:, :-1 * self.delay, :]  # truncate to text end
        reminding_taste_token_emb = taste_token_emb[:, -1 * self.delay:, :]
        # 5. step by step decode
        for text_token, taste_emb in self.inference_wrapper(lm_input, reminding_taste_token_emb, max_len, min_len, uuid, active_rule=active_rule, stop_id=stop_id):
            yield (text_token, taste_emb)

    def ban_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Ban certain tokens by setting their logits to -inf.

        Args:
            logits (torch.Tensor): Logits tensor of shape (..., vocab_size)

        Returns:
            torch.Tensor: Modified logits with banned tokens set to -inf
        """
        if len(self.banned_token_ids) > 0:
            logits[:, self.banned_token_ids] = float('-inf')
        return logits

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            ignore_eos: bool = True,
            stop_id: int = None,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.text_sampling_callable(weighted_scores)
            if (not ignore_eos) or (top_ids != stop_id):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, reminding_taste_token_emb, max_len, min_len, uuid, active_rule=None, stop_id=None):

        # active_rule is a callable with signature: active_rule(text_logp, cache=None, device=None) -> (is_active, forced_next_tokens)
        # Returns:
        #   - is_active: whether the rule should be applied
        #   - forced_next_tokens: list of token IDs to force as next tokens if active

        assert reminding_taste_token_emb.size(1) == self.delay
        if hasattr(self, 'vllm'):
            raise NotImplementedError
            
        else:
            text_out_tokens_queue = []
            cache = None

            for i in range(max_len):
                # sampling text
                hidden_pred, cache = self.slm.forward_one_step(
                    lm_input,
                    masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                    cache=cache
                )
                text_logits = self.slm.forward_lm_head(hidden_pred[:, -1])
                text_logits = self.ban_tokens(text_logits)  # Ban unwanted tokens
                text_logp = text_logits.log_softmax(dim=-1)

                if i == 0 and active_rule is not None:
                    is_active, forced_next_tokens = active_rule(text_logp, cache=cache, device=lm_input.device) 
                    if not is_active:
                        break
                    while len(forced_next_tokens) > 0:
                        top_text_ids = torch.tensor([int(forced_next_tokens.pop(0))], device=lm_input.device)
                        text_emb = self.slm.forward_embed_tokens(top_text_ids.unsqueeze(0)).float()
                        lm_input = text_emb.reshape(1, 1, -1)
                        hidden_pred, cache = self.slm.forward_one_step(
                            lm_input,
                            masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                            cache=cache
                        )
                    text_logits = self.slm.forward_lm_head(hidden_pred[:, -1])
                    text_logits = self.ban_tokens(text_logits)  # Ban unwanted tokens
                    text_logp = text_logits.log_softmax(dim=-1)

                top_text_ids = self.sampling_ids(text_logp.squeeze(dim=0), ignore_eos=(True if i < min_len else False), stop_id=stop_id)
                text_emb = self.slm.forward_embed_tokens(top_text_ids.unsqueeze(0)).float()

                # stop sampling text
                if top_text_ids == stop_id:
                    break

                # sampling taste
                if i >= self.delay:
                    if self.use_continue:
                        z, _, _ = self.out_module.predict_taste_latent(hidden_pred.float())
                        taste_emb = self.taste_stage1.neck_proj_out(z)
                    else:
                        z, _, _ = self.out_module.predict_taste_latent(hidden_pred.float())
                        vq_module = self.taste_stage1.taste_tokenizer.vq.rvq
                        taste_emb = vq_module.project_out(z)
                    yield (text_out_tokens_queue.pop(0), taste_emb)
                else:
                    taste_emb = reminding_taste_token_emb[:, i, :].unsqueeze(1)

                text_out_tokens_queue.append(top_text_ids)
                lm_input = self.fusing_module(text_emb, taste_emb, torch.tensor([1]), delay=0).reshape(1, 1, -1)

            # (reminding) sampling taste
            while len(text_out_tokens_queue) > 0:
                if self.use_continue:
                    z, _, _ = self.out_module.predict_taste_latent(hidden_pred.float())
                    taste_emb = self.taste_stage1.neck_proj_out(z)
                else:
                    z, _, _ = self.out_module.predict_taste_latent(hidden_pred.float())
                    vq_module = self.taste_stage1.taste_tokenizer.vq.rvq
                    taste_emb = vq_module.project_out(z)
                yield (text_out_tokens_queue.pop(0), taste_emb)

                text_emb = self.fusing_module.pad_text_embed.unsqueeze(0).unsqueeze(0)
                lm_input = self.fusing_module(text_emb, taste_emb, torch.tensor([1]), delay=0).reshape(1, 1, -1)
                hidden_pred, cache = self.slm.forward_one_step(
                    lm_input,
                    masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                    cache=cache
                )
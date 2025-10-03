import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Callable, Generator
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


def expand_conversations_to_messages(batch, device):
    full_text_token = batch['text_token']  # [B, L] - conversations with padded text tokens
    full_audio_feature = batch['audio_feature'].to(dtype=torch.float16)  # [B, M, L, D] - convert to float16 early
    full_audio_feature_lens = batch['audio_feature_lens']  # [B, M] - lengths of audio features for each message
    full_token_message_ids = batch['token_message_ids']  # [B, L] - message IDs for each token

    # Expand all messages from B,M,L,D to new batch format
    new_batch_text_tokens = []
    new_batch_text_lens = []
    new_batch_audio_features = []
    new_batch_audio_lens = []

    # Track mapping for reconstruction - (original_batch_idx, message_id, new_batch_position)
    message_mapping = []

    B, M, L, D = full_audio_feature.size()
    new_batch_idx = 0

    for batch_idx in range(B):
        text_tokens = full_text_token[batch_idx]  # [L]
        token_msg_ids = full_token_message_ids[batch_idx]  # [L]
        audio_features = full_audio_feature[batch_idx]  # [M, L, D]
        audio_lens = full_audio_feature_lens[batch_idx]  # [M]

        # Find unique message IDs (excluding -1 for special tokens/padding)
        unique_msg_ids = torch.unique(token_msg_ids)
        unique_msg_ids = unique_msg_ids[unique_msg_ids != -1]  # Remove -1
        unique_msg_ids = sorted(unique_msg_ids.tolist())  # Sort for consistent ordering

        for msg_id in unique_msg_ids:
            # Extract text tokens for this message
            text_mask = (token_msg_ids == msg_id)
            msg_text_tokens = text_tokens[text_mask]

            # Get audio features for this message (msg_id corresponds to message index)
            if msg_id < M and audio_lens[msg_id] > 0:
                msg_audio_len = audio_lens[msg_id].item()
                msg_audio_features = audio_features[msg_id, :msg_audio_len, :]  # [audio_len, D]

                # Only add if both text and audio exist for this message
                if msg_text_tokens.size(0) > 0 and msg_audio_features.size(0) > 0:
                    new_batch_text_tokens.append(msg_text_tokens)
                    new_batch_text_lens.append(msg_text_tokens.size(0))
                    new_batch_audio_features.append(msg_audio_features)
                    new_batch_audio_lens.append(msg_audio_len)

                    # Store mapping: (original_batch_idx, message_id, new_batch_position)
                    message_mapping.append((batch_idx, msg_id, new_batch_idx))
                    new_batch_idx += 1

    # Create new batch format for stage1 processing
    new_text_token = pad_sequence(new_batch_text_tokens, batch_first=True, padding_value=0).to(device)
    new_text_token_len = torch.tensor(new_batch_text_lens, dtype=torch.int32, device=device).to(device)
    # Ensure audio features match the conv1 layer dtype (float16) and move to correct device
    new_audio_feature = pad_sequence(new_batch_audio_features, batch_first=True, padding_value=0)
    target_dtype = torch.float16
    new_audio_feature = new_audio_feature.to(dtype=target_dtype, device=device).to(device)
    new_audio_feature_len = torch.tensor(new_batch_audio_lens, dtype=torch.int32, device=device).to(device)

    return new_text_token, new_text_token_len, new_audio_feature, new_audio_feature_len, message_mapping
    
    
    
def reconstruct_conversations_from_messages(new_taste_embs, new_taste_latents, message_mapping,
                     full_text_token, full_text_token_len, full_token_message_ids,
                     new_text_token_len, device):
    """Reconstruct taste_token_emb and taste_latent using saved mapping

    Args:
        new_taste_embs: Processed taste embeddings from expanded batch [num_messages, msg_len, emb_dim]
        new_taste_latents: Processed taste latents from expanded batch [num_messages, msg_len, latent_dim]
        message_mapping: List of tuples (original_batch_idx, message_id, new_batch_position)
        full_text_token: Original text tokens [B, L]
        full_text_token_len: Original text lengths [B]
        full_token_message_ids: Message IDs for each token [B, L]
        new_text_token_len: Lengths of expanded messages [num_messages]
        device: Target device

    Returns:
        Tuple of (text_token, text_token_len, taste_token_emb, taste_latent)
    """
    # Reconstruct taste_token_emb and taste_latent using saved mapping
    reconstructed_taste_embs = []
    reconstructed_taste_latents = []

    B = full_text_token.size(0)

    for batch_idx in range(B):
        text_len = full_text_token_len[batch_idx].item()
        padded_text_len = full_text_token.size(1)  # Use padded length instead
        token_msg_ids = full_token_message_ids[batch_idx]  # [L]
        emb_dim = new_taste_embs.size(-1)
        latent_dim = new_taste_latents.size(-1)

        # Initialize with zeros (for special tokens with msg_id == -1) - use padded length
        conversation_taste_emb = torch.zeros(padded_text_len, emb_dim, device=device)
        conversation_taste_latent = torch.zeros(padded_text_len, latent_dim, device=device)

        # Fill in processed embeddings using the mapping
        for orig_batch_idx, msg_id, new_batch_pos in message_mapping:
            if orig_batch_idx == batch_idx:
                # Find positions where this message appears in the original conversation
                msg_positions = (token_msg_ids[:text_len] == msg_id).nonzero(as_tuple=True)[0]

                # Get the processed embeddings from the new batch
                msg_taste_emb = new_taste_embs[new_batch_pos]  # [msg_len, emb_dim]
                msg_taste_latent = new_taste_latents[new_batch_pos]  # [msg_len, latent_dim]
                msg_len = new_text_token_len[new_batch_pos].item()

                # Map processed embeddings to conversation positions
                min_len = min(len(msg_positions), msg_len)
                if min_len > 0:
                    conversation_taste_emb[msg_positions[:min_len]] = msg_taste_emb[:min_len]
                    conversation_taste_latent[msg_positions[:min_len]] = msg_taste_latent[:min_len]

        reconstructed_taste_embs.append(conversation_taste_emb)
        reconstructed_taste_latents.append(conversation_taste_latent)

    # Convert to batch tensor (already same length as padded text tokens)
    taste_token_emb = torch.stack(reconstructed_taste_embs, dim=0)
    taste_latent = torch.stack(reconstructed_taste_latents, dim=0)

    return full_text_token.to(device), full_text_token_len.to(device), taste_token_emb, taste_latent


def apply_template_on_message(tokenizer, role, asr_text, stream = False):
    """
    Package text into ChatML format with message dimension tracking.

    Args:
        tokenizer: Tokenizer instance for encoding text
        role: Role ('system', 'user', 'assistant') for this turn
        asr_text: Text content for the message

    Returns:
        formatted_text_token: tensor of shape [1, seq_len]
        formatted_text_token_len: tensor with sequence length
        token_message_ids: list tracking which tokens are content (0) vs structural (-1)
    """
    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"
    newline_token = "\n"

    # Encode each component
    im_start_id = tokenizer.encode(im_start_token, add_special_tokens=False)
    im_end_id = tokenizer.encode(im_end_token, add_special_tokens=False)
    newline_id = tokenizer.encode(newline_token, add_special_tokens=False)
    role_ids = tokenizer.encode(role, add_special_tokens=False)
    content_tokens = tokenizer.encode(asr_text, add_special_tokens=False)

    # Build concatenated tokens with message dimension tracking
    concatenated_tokens = []
    token_message_ids = []

    # Define token groups with their corresponding message IDs
    if stream == True: 
        token_groups = [
            (im_start_id, -1),
            (role_ids, -1),
            (newline_id, -1),
            (content_tokens, 0),
        ]
    else:
       token_groups = [
            (im_start_id, -1),
            (role_ids, -1),
            (newline_id, -1),
            (content_tokens, 0),
            (newline_id, -1),
            (im_end_id, -1)
        ]
    # Iterate over the groups and extend tokens and message IDs
    for tokens, message_id in token_groups:
        concatenated_tokens.extend(tokens)
        token_message_ids.extend([message_id] * len(tokens))

    # Convert to tensors
    formatted_text_token = torch.tensor(concatenated_tokens, dtype=torch.int32).unsqueeze(0)
    formatted_text_token_len = torch.tensor([len(concatenated_tokens)], dtype=torch.int32)

    return formatted_text_token, formatted_text_token_len, token_message_ids



def extract_taste_from_message(
    slm,
    taste_token_emb: torch.Tensor,
    formatted_text_token: torch.Tensor,
    formatted_text_token_len: torch.Tensor,
    token_message_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct taste embeddings for SFT mode.

    Args:
        slm: Speech Language Model with forward_embed_tokens method
        taste_token_emb: Original taste token embeddings
        formatted_text_token: Formatted text tokens for SFT mode
        formatted_text_token_len: Length of formatted text tokens
        token_message_ids: Message IDs for each token position

    Returns:
        Tuple of (text_token_emb, taste_token_emb, text_token, text_token_len)
    """
    text_token_emb = slm.forward_embed_tokens(formatted_text_token).float()
    # SFT mode: reconstruct like training _prepare_for_sft_training
    device = formatted_text_token.device
    formatted_len = formatted_text_token_len.item()
    emb_dim = taste_token_emb.size(-1)

    # Initialize with all zeros for formatted sequence
    reconstructed_taste_emb = torch.zeros(1, formatted_len, emb_dim, device=device)

    # Find content positions (where token_message_ids == 0)
    content_mask = (token_message_ids.squeeze() == 0)[:formatted_len]

    if content_mask.any():
        content_positions = content_mask.nonzero(as_tuple=True)[0]
        min_len = min(len(content_positions), taste_token_emb.size(1))
        if min_len > 0:
            reconstructed_taste_emb[0, content_positions[:min_len]] = taste_token_emb[0, :min_len]

    taste_token_emb = reconstructed_taste_emb.to(device)
    # Use formatted tokens for text processing
    text_token = formatted_text_token
    text_token_len = formatted_text_token_len

    return text_token_emb, taste_token_emb, text_token, text_token_len
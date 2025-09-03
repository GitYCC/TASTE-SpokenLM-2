"""
Streaming generation for TASTE-SpokenLM with Iterator pattern and threading support.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Iterator, Optional, Callable, TYPE_CHECKING
from threading import Thread
from queue import Queue, Empty
import time

from .segment import TASTESegment, StreamingResult
from .conversation import build_conditional_prompt, check_completion_status, SPECIAL_TOKENS

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


class StreamingGenerationState:
    """Encapsulates the state for streaming generation."""
    
    def __init__(self, device, llm_embed_tokens, generated_text_tokens=None, generated_taste_tokens=None):
        self.device = device
        self.llm_embed_tokens = llm_embed_tokens
        self.generated_text_tokens = generated_text_tokens or []
        self.generated_taste_tokens = generated_taste_tokens or []
        self.generation_step = 0
        
    def add_text_token(self, text_id, action):
        """Add text token if action is valid."""
        if action not in ('wait_for_taste', 'terminate'):
            self.generated_text_tokens.append(text_id)
            
    def add_taste_token(self, taste_ids, taste_action):
        """Add taste token if sampled."""
        if taste_action == 'sample':
            self.generated_taste_tokens.append(taste_ids.squeeze(0).squeeze(0))
            
    def has_generated_tokens(self):
        """Check if any tokens have been generated."""
        return bool(self.generated_text_tokens)
        
    def increment_step(self):
        """Increment generation step counter."""
        self.generation_step += 1
        

def _get_model_components(model):
    """Extract model components following the generate method pattern."""
    if model.spoken_lm._use_lora:
        base = model.spoken_lm.language_model.base_model.model
    else:
        base = model.spoken_lm.language_model

    return {
        'llm_embed_tokens': base.model.embed_tokens,
        'llm_backbone': base.model,
        'lm_head': base.lm_head,
        'device': base.device,
        'vq_module': model.audio_tower.vq.rvq
    }


def _setup_generation_state(model, processor, input_segments, text_top_p, taste_top_p, 
                           text_temperature, repetition_penalty, eos_id):
    """Setup initial generation state and model configuration."""
    components = _get_model_components(model)
    device = components['device']
    
    # Build conditional prompt
    prompt_text_ids, prompt_taste_ids = build_conditional_prompt(
        input_segments, processor.llm_tokenizer
    )
    prompt_text_ids = prompt_text_ids.to(device)
    prompt_taste_ids = prompt_taste_ids.to(device)
    
    # Register and reset taste sampler
    model.spoken_lm.register_taste_sampler(
        llm_tokenizer=processor.llm_tokenizer,
        text_top_p=text_top_p,
        taste_top_p=taste_top_p,
        text_temperature=text_temperature,
        repetition_penalty=repetition_penalty,
    )
    model.spoken_lm.taste_sampler.reset(has_prefix=True, stop_id=eos_id)
    
    # Initialize generation state
    inputs_embeds = components['llm_embed_tokens'](prompt_text_ids)
    input_ids = prompt_text_ids.clone()
    state = StreamingGenerationState(device, components['llm_embed_tokens'])
    
    return components, inputs_embeds, input_ids, state


def _perform_forward_pass(llm_backbone, lm_head, model, inputs_embeds, vq_module):
    """Perform forward pass through the model."""
    llm_outputs = llm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        output_hidden_states=True,
        return_dict='pt'
    )
    
    text_logits = lm_head(llm_outputs.last_hidden_state)
    taste_logits, _ = model.spoken_lm.extract_for_bridge_out_llm(llm_outputs, vq_module)
    
    return text_logits, taste_logits


def _update_inputs_embeds(model, llm_embed_tokens, text_id, taste_ids, taste_action, 
                         inputs_embeds, device, vq_module):
    """Construct and update inputs_embeds for next iteration."""
    if taste_action == 'sample':
        # Use sampled taste for audio embedding
        last_asr_embed = model.spoken_lm.encode_audio(taste_ids, vq_module)
        new_inputs_embeds = model.spoken_lm.fuse_for_bridge_in_llm(
            llm_embed_tokens.weight[text_id].reshape(1, 1, -1),
            last_asr_embed
        )
    else:
        # Use padding for non-taste tokens
        new_inputs_embeds = model.spoken_lm.fuse_for_bridge_in_llm(
            llm_embed_tokens.weight[text_id].reshape(1, 1, -1),
            model.spoken_lm.pad_audio_unit_embed.reshape(1, 1, -1)
        )
    
    # Update inputs_embeds with proper dtype
    llm_dtype = next(llm_embed_tokens.parameters()).dtype
    return torch.cat([inputs_embeds, new_inputs_embeds], dim=1).to(dtype=llm_dtype, device=device)


def _create_segment_from_tokens(text_tokens, taste_tokens, device, is_complete, reason):
    """Create TASTESegment and result from generated tokens."""
    if not text_tokens:
        return None
        
    gen_text_ids = torch.tensor([text_tokens], device=device, dtype=torch.long)
    
    if taste_tokens:
        gen_taste_ids = torch.stack(taste_tokens).unsqueeze(0)
    else:
        gen_taste_ids = torch.empty(1, len(text_tokens), 4, device=device, dtype=torch.long)
    
    segment = TASTESegment(
        role='assistant',
        modality='audio',
        text_ids=gen_text_ids,
        taste_ids=gen_taste_ids
    )
    
    return {
        'is_complete': is_complete,
        'completion_reason': reason,
        'segment': segment
    }


def _should_emit_segment(completion_status, should_emit_segment, text_id, action, 
                        generated_text_tokens):
    """Determine if we should emit a segment."""
    if completion_status.is_complete:
        return True
    elif should_emit_segment is not None:
        return should_emit_segment(text_id, action, completion_status)
    elif completion_status.reason == 'segment_end':
        return True
    else:
        # Emit periodically for streaming experience (every ~10 tokens)
        return len(generated_text_tokens) % 10 == 0 and len(generated_text_tokens) > 0


def streaming_generate(
    model: "TasteForCausalLM",
    processor: "TasteProcessor", 
    input_segments: List[TASTESegment],
    text_top_p: float = 0.3,
    taste_top_p: float = 0.0,
    text_temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    max_length: int = 512,
    eos_id: Optional[int] = None,
    should_emit_segment: Optional[Callable] = None
) -> Iterator[Dict]:
    """
    Streaming generation of conversation responses using Iterator pattern with threading.
    
    Generates multimodal conversation responses incrementally, yielding results as they
    become available. Uses background threading to avoid blocking the main thread.
    
    Args:
        model: TasteForCausalLM model for generation
        processor: TasteProcessor for tokenization 
        input_segments: List of input conversation segments
        text_top_p: Top-p sampling for text generation (default: 0.3)
        taste_top_p: Top-p sampling for TASTE tokens (default: 0.0, greedy)
        text_temperature: Temperature for text sampling (default: 0.5) 
        repetition_penalty: Penalty for repeated tokens (default: 1.1)
        max_length: Maximum generation length (default: 512)
        eos_id: End-of-sequence token ID (optional)
        should_emit_segment: Function to determine when to emit segments (optional)
        
    Yields:
        Dict: StreamingResult with keys:
            - is_complete: bool indicating if generation is finished
            - completion_reason: str reason for completion
            - segment: TASTESegment containing generated content
    """
    # Set default eos_id to segment_end token if not provided
    if eos_id is None:
        segment_end_token = SPECIAL_TOKENS['segment_end']
        segment_end_ids = processor.llm_tokenizer.encode(segment_end_token, add_special_tokens=False)
        if segment_end_ids:
            eos_id = segment_end_ids[0]   

    # Setup result queue for thread communication
    result_queue = Queue()
    exception_queue = Queue()
    
    def _generation_worker():
        """Background worker thread for generation."""
        try:
            # Setup initial generation state
            components, inputs_embeds, input_ids, state = _setup_generation_state(
                model, processor, input_segments, text_top_p, taste_top_p, 
                text_temperature, repetition_penalty, eos_id
            )
            
            device = components['device']
            llm_embed_tokens = components['llm_embed_tokens']
            llm_backbone = components['llm_backbone']
            lm_head = components['lm_head']
            vq_module = components['vq_module']
            
            with torch.no_grad():
                # Main autoregressive generation loop
                while state.generation_step < max_length:
                    # Forward pass
                    text_logits, taste_logits = _perform_forward_pass(
                        llm_backbone, lm_head, model, inputs_embeds, vq_module
                    )
                    
                    # Sample next tokens
                    text_id, taste_ids, action, taste_action = model.spoken_lm.taste_sampler.update(
                        text_logits, taste_logits, input_ids=input_ids
                    )
                    
                    # Update input_ids
                    input_ids = torch.nn.functional.pad(input_ids, (0, 1), 'constant', text_id)
                    
                    # Update inputs_embeds for next iteration
                    inputs_embeds = _update_inputs_embeds(
                        model, llm_embed_tokens, text_id, taste_ids, taste_action, 
                        inputs_embeds, device, vq_module
                    )
                    
                    # Track generated tokens
                    state.add_text_token(text_id, action)
                    state.add_taste_token(taste_ids, taste_action)
                    
                    # Check completion status
                    completion_status = check_completion_status(text_id, action)
                    
                    # Handle termination
                    if action == 'terminate':
                        if state.has_generated_tokens():
                            result = _create_segment_from_tokens(
                                state.generated_text_tokens, state.generated_taste_tokens,
                                device, True, 'terminate'
                            )
                            result_queue.put(result)
                        break
                    
                    # Check if we should emit a segment
                    should_emit = _should_emit_segment(
                        completion_status, should_emit_segment, text_id, action,
                        state.generated_text_tokens
                    )
                    
                    # Emit segment if needed
                    if should_emit and state.has_generated_tokens():
                        result = _create_segment_from_tokens(
                            state.generated_text_tokens, state.generated_taste_tokens,
                            device, completion_status.is_complete, completion_status.reason
                        )
                        result_queue.put(result)
                        
                        # Break if complete
                        if completion_status.is_complete:
                            break
                    
                    state.increment_step()
                
                # Final emission if needed
                if not completion_status.is_complete and state.has_generated_tokens():
                    result = _create_segment_from_tokens(
                        state.generated_text_tokens, state.generated_taste_tokens,
                        device, True, 'max_length'
                    )
                    result_queue.put(result)
        
        except Exception as e:
            exception_queue.put(e)
    
    # Start background generation thread and consume results
    return _run_streaming_generation_thread(_generation_worker, result_queue, exception_queue)


def _run_streaming_generation_thread(worker_func, result_queue, exception_queue):
    """Run worker thread and yield streaming results."""
    worker_thread = Thread(target=worker_func)
    worker_thread.daemon = True
    worker_thread.start()
    
    try:
        while True:
            # Check for exceptions first
            if not exception_queue.empty():
                raise exception_queue.get_nowait()
            
            try:
                # Try to get result with timeout
                result = result_queue.get(timeout=1.0)
                yield result
                
                # If this result is complete, we're done
                if result['is_complete']:
                    break
                    
            except Empty:
                # Check if thread is still alive
                if not worker_thread.is_alive():
                    # Thread died without putting a result - check for exception
                    if not exception_queue.empty():
                        raise exception_queue.get_nowait()
                    else:
                        # Thread finished without error but no final result
                        break
                # Continue waiting for results
                continue
    finally:
        # Ensure thread cleanup
        if worker_thread.is_alive():
            worker_thread.join(timeout=1.0)
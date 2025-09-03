
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class TasteAudioTowerConfig(PretrainedConfig):
    model_type = "taste"
    is_composition = True

    def __init__(
        self,
        encoder_input_size=512,
        text_token_size=51866,
        audio_embed_dim=1280,  # useless
        quantization_on=False,
        is_joint_encoder_segmenter=True,
        audio_dropout_ratio=0.0,
        kwargs_for_joint_encoder_segmenter=None,
        kwargs_for_quantizer=None,
        encoder__target_hidden_layer=6,
        encoder__unfreeze_hidden_layers_from_last=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder_input_size = encoder_input_size
        self.text_token_size = text_token_size
        self.audio_embed_dim = audio_embed_dim
        self.quantization_on = quantization_on
        self.is_joint_encoder_segmenter = is_joint_encoder_segmenter
        self.audio_dropout_ratio = audio_dropout_ratio
        self.encoder__target_hidden_layer= encoder__target_hidden_layer
        self.encoder__unfreeze_hidden_layers_from_last = encoder__unfreeze_hidden_layers_from_last
        self.kwargs_for_joint_encoder_segmenter = kwargs_for_joint_encoder_segmenter
        self.kwargs_for_quantizer = kwargs_for_quantizer


class TasteSpeechDecoderConfig(PretrainedConfig):
    model_type = "taste"
    is_composition = True

    def __init__(
        self,
        encoder_input_size=512,
        audio_encoder_input_size=-1,
        llm_input_size=1024,
        llm_output_size=1024,
        text_token_size=51866,
        speech_token_size=4096,
        length_normalized_loss=True,
        lsm_weight=0.0,
        spk_embed_dim=192,
        fuse_encoded_audio_text_type='concat',
        fuse_encoded_audio_text_kwargs={},
        skip_prefix_idx=0,
        encoder__attention_heads=8,
        encoder__linear_units=2048,
        encoder__num_blocks=3,
        encoder__dropout_rate=0.1,
        encoder__positional_dropout_rate=0.1,
        encoder__attention_dropout_rate=0,
        encoder__normalize_before=True,
        encoder__input_layer='linear',
        encoder__pos_enc_layer_type='rel_pos_espnet',
        encoder__selfattention_layer_type='rel_selfattn',
        encoder__use_cnn_module=False,
        encoder__macaron_style=False,
        encoder__use_dynamic_chunk=False,
        encoder__use_dynamic_left_chunk=False,
        encoder__static_chunk_size=1,
        llm__attention_heads=8,
        llm__linear_units=2048,
        llm__num_blocks=7,
        llm__dropout_rate=0.1,
        llm__positional_dropout_rate=0.1,
        llm__attention_dropout_rate=0,
        llm__input_layer='linear_legacy',
        llm__pos_enc_layer_type='rel_pos_espnet',
        llm__selfattention_layer_type='rel_selfattn',
        llm__static_chunk_size=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder_input_size = encoder_input_size
        self.audio_encoder_input_size = audio_encoder_input_size
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.text_token_size = text_token_size
        self.speech_token_size = speech_token_size
        self.length_normalized_loss = length_normalized_loss
        self.lsm_weight = lsm_weight
        self.spk_embed_dim = spk_embed_dim
        self.fuse_encoded_audio_text_type = fuse_encoded_audio_text_type
        self.fuse_encoded_audio_text_kwargs = fuse_encoded_audio_text_kwargs
        self.skip_prefix_idx = skip_prefix_idx
        self.encoder__attention_heads = encoder__attention_heads
        self.encoder__linear_units = encoder__linear_units
        self.encoder__num_blocks = encoder__num_blocks
        self.encoder__dropout_rate = encoder__dropout_rate
        self.encoder__positional_dropout_rate = encoder__positional_dropout_rate
        self.encoder__attention_dropout_rate = encoder__attention_dropout_rate
        self.encoder__normalize_before = encoder__normalize_before
        self.encoder__input_layer = encoder__input_layer
        self.encoder__pos_enc_layer_type = encoder__pos_enc_layer_type
        self.encoder__selfattention_layer_type = encoder__selfattention_layer_type
        self.encoder__use_cnn_module = encoder__use_cnn_module
        self.encoder__macaron_style = encoder__macaron_style
        self.encoder__use_dynamic_chunk = encoder__use_dynamic_chunk
        self.encoder__use_dynamic_left_chunk = encoder__use_dynamic_left_chunk
        self.encoder__static_chunk_size = encoder__static_chunk_size
        self.llm__attention_heads = llm__attention_heads
        self.llm__linear_units = llm__linear_units
        self.llm__num_blocks = llm__num_blocks
        self.llm__dropout_rate = llm__dropout_rate
        self.llm__positional_dropout_rate = llm__positional_dropout_rate
        self.llm__attention_dropout_rate = llm__attention_dropout_rate
        self.llm__input_layer = llm__input_layer
        self.llm__pos_enc_layer_type = llm__pos_enc_layer_type
        self.llm__selfattention_layer_type = llm__selfattention_layer_type
        self.llm__static_chunk_size = llm__static_chunk_size


class TasteSpokenLMConfig(PretrainedConfig):
    model_type = "taste"
    is_composition = True

    def __init__(
        self,
        sos_id=128000,
        loss_weights='0.05-0.3-0.3-0.2-0.15',
        delay=3,
        delay_level='word',
        audio_embed_conv_mode='fill_forward',
        in_llm_module='weighted_sum',
        out_llm_module='continue_latent_linear_last',
        use_lora=True,
        kwargs_for_lora=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sos_id = sos_id
        self.loss_weights = loss_weights
        self.delay = delay
        self.delay_level = delay_level
        self.audio_embed_conv_mode = audio_embed_conv_mode
        self.in_llm_module = in_llm_module
        self.out_llm_module = out_llm_module
        self.use_lora = use_lora
        self.kwargs_for_lora = kwargs_for_lora


class TasteConfig(PretrainedConfig):
    model_type = "taste"
    is_composition = True

    def __init__(
        self,
        audio_tower_config=None,
        speech_decoder_config=None,
        spoken_lm_config=None,
        text_config=None,
        asr_config=None,
        ignore_index=-100,
        load_language_model_during_init=True,
        **kwargs,
    ):

        if audio_tower_config is None:
            audio_tower_config = TasteAudioTowerConfig()

        if speech_decoder_config is None:
            speech_decoder_config = TasteSpeechDecoderConfig()

        if spoken_lm_config is None:
            spoken_lm_config = TasteSpokenLMConfig()

        self.audio_tower_config = audio_tower_config
        self.speech_decoder_config = speech_decoder_config
        self.spoken_lm_config = spoken_lm_config

        self.ignore_index = ignore_index
        self.load_language_model_during_init=load_language_model_during_init

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        if isinstance(asr_config, dict):
            asr_config["model_type"] = asr_config["model_type"] if "model_type" in asr_config else "whisper"
            asr_config = CONFIG_MAPPING[asr_config["model_type"]](**asr_config)
        elif asr_config is None:
            asr_config = CONFIG_MAPPING["whisper"]()

        self.asr_config = asr_config

        super().__init__(**kwargs)

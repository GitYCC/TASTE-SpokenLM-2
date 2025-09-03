
import re

import torch
import torch.nn.functional as F

from taste_speech.modules_taste.cosyvoice.utils import IGNORE_ID


class TasteSampler:
    def __init__(self, delay, delay_level, text_vocab_size, llm_tokenizer,
                text_top_p=0.0, taste_top_p=0.0, text_temperature=1.0, repetition_penalty=1.0):
        if delay == 0:
            raise NotImplementedError()

        self.llm_tokenizer = llm_tokenizer
        self._register_llm_word_start_tokens(text_vocab_size, llm_tokenizer)
        self._register_baning_ids(text_vocab_size, llm_tokenizer)
        self._register_sentence_end_tokens(text_vocab_size, llm_tokenizer)

        self.delay = delay
        self.delay_level = delay_level

        self.text_top_p = text_top_p
        self.taste_top_p = taste_top_p
        self.text_temperature = text_temperature
        self.repetition_penalty = repetition_penalty

        self._ban_value = -100000

    def _register_sentence_end_tokens(self, text_vocab_size, llm_tokenizer):
        self.sentance_end_set = set()
        for i in range(text_vocab_size):
            _subword = llm_tokenizer.decode(i)
            if '.' in _subword:
                self.sentance_end_set.add(i)

    def _register_llm_word_start_tokens(self, text_vocab_size, llm_tokenizer):
        self.word_start_set = set()
        for i in range(text_vocab_size):
            if i >= 128000: 
                self.word_start_set.add(i)
                continue
            _subword = llm_tokenizer.decode(i)
            if _subword[0] == ' ':
                self.word_start_set.add(i)

    def _register_baning_ids(self, text_vocab_size, llm_tokenizer):
        self.ban_ids = []
        for i in range(text_vocab_size):
            _subword = llm_tokenizer.decode(i)
            if _subword == '.':
                continue
            if not re.search(r'[.,\'a-zA-Z0-9]', _subword):
                self.ban_ids.append(i)
            elif '\n' in _subword:
                self.ban_ids.append(i)
        self.ban_ids.append(128001)  # end of sentence

    def _top_p_filtering(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = self._ban_value
        return logits
    
    def _apply_repetition_penality(self, input_ids, logits):
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
        logits_processed = logits.scatter(1, input_ids, score)
        return logits_processed

    def reset(self, extra_words, has_prefix=True, stop_id=None):
        self.word_start_history = []
        self._end_countdown = None
        self._extra_words = extra_words
        self._max_words = extra_words * 3
        self._end_text_sampling = False
        self._word_counter = 0
        self.has_prefix = has_prefix
        self.stop_id = stop_id

    def text_sample(self, text_logits, text_top_p, input_ids):
        greedy = text_top_p == 0.0

        text_logits[..., self.ban_ids] = self._ban_value
        if greedy:
            next_text_input_id = text_logits[:, -1:, :].argmax(-1).item()
        else:
            text_logits = self._top_p_filtering(text_logits[:, -1, :], top_p=text_top_p)
            text_logits = self._apply_repetition_penality(input_ids, text_logits)
            text_logits /= self.text_temperature
            next_text_input_id = torch.multinomial(F.softmax(text_logits, dim=-1), num_samples=1).item()

        return next_text_input_id

    def taste_sample(self, taste_logits, taste_top_p):
        greedy = taste_top_p == 0.0

        if greedy:
            next_taste_ids = taste_logits[:, -1:, :, :].argmax(-1)
        else:
            next_taste_ids = []
            for i in range(4):
                tmp_logits = self._top_p_filtering(taste_logits[:, -1, i, :], top_p=taste_top_p)
                next_taste_ids.append(torch.multinomial(F.softmax(tmp_logits, dim=-1), num_samples=1).item())
            next_taste_ids = torch.tensor([[next_taste_ids]], device=taste_logits.device, dtype=torch.int64)

        return next_taste_ids

    def update(self, text_logits, taste_logits, input_ids):
        text_id = self.text_sample(text_logits, self.text_top_p, input_ids)

        # when text sampling ends, start waiting for taste token sampling
        is_wait_for_taste = self._end_text_sampling

        # check if locating at word-start and update self._end_countdown
        if self._word_counter == 0: # begin
            is_word_start = True
        elif is_wait_for_taste: # wait for taste
            if self._end_countdown is None:
                self._end_countdown = self.delay
            self._end_countdown -= 1

            is_word_start = True
        else:
            is_word_start = text_id in self.word_start_set

        # update history of is_word_start
        self.word_start_history.append(is_word_start)

        # check if text sampling end
        if ((self._word_counter >= self._extra_words) and (text_id in self.sentance_end_set)) \
                    or (self._word_counter >= self._max_words):
            self._end_text_sampling = True

        # stop (not include the stop_id)
        if self.stop_id is not None and text_id == self.stop_id:
            self._end_text_sampling = True
            self._end_countdown = self.delay - 1
            is_wait_for_taste = True

        # update word_counter
        if is_word_start:
            self._word_counter += 1

        # determine action
        is_end = (self._end_countdown == 0)
        if is_end:
            action = 'terminate'
        elif is_wait_for_taste:
            action = 'wait_for_taste'
        elif is_word_start:
            action = 'continue_at_word_start'
        else:
            action = 'continue_not_at_word_start'

        # check if sampling taste
        if self.delay_level == 'token':
            is_started_sampling_taste =  (len(self.word_start_history) > self.delay)
            is_taste_sampling = is_started_sampling_taste and self.word_start_history[-1-self.delay]
        elif self.delay_level == 'word':
            is_started_sampling_taste = (sum([1 if s else 0 for s in self.word_start_history]) > self.delay)
            is_taste_sampling = is_started_sampling_taste and is_word_start

        # sample taste_ids
        if is_taste_sampling:
            taste_ids = self.taste_sample(taste_logits, taste_top_p=self.taste_top_p)
        else:
            taste_ids = torch.tensor([[[IGNORE_ID] * 4]], device=taste_logits.device, dtype=torch.int64)

        # determine taste_action
        if is_started_sampling_taste:
            taste_action = 'sample'
        elif self.has_prefix:
            if self.delay_level == 'word':
                if is_word_start:
                    taste_action = 'use_prefix'
                else:
                    taste_action = 'use_prefix_ignore'
            elif self.delay_level == 'token':
                taste_action = 'use_prefix'
        else:
            taste_action = 'wait'

        return (text_id, taste_ids, action, taste_action)

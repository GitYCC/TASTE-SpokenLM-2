
system segment: `<|im_start|>System\n...<|im_end|>`  
user segment: `<|im_start|>User\n...<|im_end|>`  
assistant segment: `<|im_start|>Assistant\n...<|im_end|>`  

Only single system segment locate at head.  
Then ordering: assistant(with 開頭詞) -> user -> assistant -> user -> ...  

開頭詞 with speech: "Hello, how can I help you today?"  

token_id of `<|im_start|>` is 151644  
token_id of `<|im_end|>` is 151645  

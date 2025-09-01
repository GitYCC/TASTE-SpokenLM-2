
```parquet
[
  {
    'idx': 'xxx',
    'meta': {
      'scenario': ...
      ....
    },
    'message': [
      {
        'role': 'system',
        'text': 'SYSTEM PROMPT'
      },
      {
        'role': 'user',
        'text': '....',
        'audio': ~byte~,
        'timestamp_range': [~milsec~, ~milsec~]
      },
      {
        'role': 'assistant',
        'text': '....',
        'audio': ~byte~,
        'timestamp_range': [~milsec~, ~milsec~]
      },
      ....
    ]
  },
  ...
  
]
```

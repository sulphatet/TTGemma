# Tokenizer

### Design

* **Type:** SentencePiece Unigram
* **Vocabulary Size:** 64,000
* **Training Subset:** 120MB sampled in round-robin across EN/HI/NE for memory stability.

### Special Tokens

* **Language Tags:** `<eng>`, `<hin>`, `<nep>`
* **Entities:** `<URL>`, `<EMAIL>`, `<DATE>`, `<NUM>`
* **Standard:** `<s>`, `</s>`, `<pad>`, `<unk>`

### Indic Pre-tokenization

Optional pre-tokenization step designed to handle Devanagari-specific punctuation and spacing.

### Evaluation

| Metric             | English | Hindi  | Nepali |
| ------------------ | ------- | ------ | ------ |
| Bytes per Token    | 4.28    | 9.61   | 11.80  |
| Byte Fallback Rate | 0.0024  | 0.0023 | 0.0003 |
| Mean Tokens / Doc  | 572.7   | 532.2  | 423.3  |
| p99 Tokens / Doc   | 4096    | 4096   | 4096   |


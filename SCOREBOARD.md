# Project Scoreboard

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Small model, base | 29.32 | 15.0% (4.5/30) | 25.0% (1/4) | 0.236 | GPT-2 small (124M), greedy decoding, max_new_tokens=200 |
| + continued pretraining | 16.53 | 10.0% (3.0/30) | 0.0% (0/4) | 0.482 | 3 epochs, lr=5e-5, context=512 (42 docs: 2,670 train / 260 val tokens) |
| + SFT | 18.84* | 53.3% (16.0/30) | 100.0% (4/4) | 0.394 | 488 pairs (438 train / 50 val), 3 epochs, lr=5e-5, answer-only loss |

*\*Evaluated on the same held-out CPT validation block for reference; SFT pairs use the `<|user|> ... <|assistant|> ... <|end|>` chat template.*

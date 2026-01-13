## Speculative Decoding Inference Engine
A custom transformer implementation that accelerates LLM inference using blockwise parallel decoding.

Implementation & results:
* Developed decoder transformer from scratch, with several heads to predict multiple future tokens simultaneously
* Froze pre-trained model weights to fine-tune only the speculative decoding heads
* Validates drafted sequences in a single forward pass instead of serial decoding steps
* Achieves 1.4x wall-clock speedup on the TinyStories dataset with lossless accuracy

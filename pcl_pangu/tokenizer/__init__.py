import os
current_dir = os.path.dirname(os.path.abspath(__file__))

vocab_4w = os.path.join(current_dir, "bpe_4w_pcl/vocab")
vocab_13w = os.path.join(current_dir, "spm_13w/spm.128k.model.1")

__all__ = [
    "vocab_4w",
    "vocab_13w"
]
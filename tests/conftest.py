"""Shared fixtures: a tiny randomly initialised causal LM, built in code.

No downloads, CPU only. Qwen2 is used so the tests exercise the same model class
(and the same ``logits_to_keep`` / ``position_ids`` handling) as the real run.
"""

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB = 48
PAD = 0
EOS = 1


def build_tiny_model(seed: int = 0, vocab: int = VOCAB) -> nn.Module:
    torch.manual_seed(seed)
    cfg = Qwen2Config.from_dict(
        {
            "vocab_size": vocab,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
            "pad_token_id": PAD,
            "eos_token_id": EOS,
            "bos_token_id": None,
        }
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture
def tiny_model() -> nn.Module:
    return build_tiny_model()


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(1234)

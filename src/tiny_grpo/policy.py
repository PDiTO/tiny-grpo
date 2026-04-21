"""Everything that touches the language model: loading, prompting, sampling and
scoring completions.

Layout of a rollout batch, for ``B`` completions::

    prompt_ids      (B, P)  left padded, so every prompt ends at column P - 1
    completion_ids  (B, T)  right padded; tokens after the first EOS are padding
    completion_mask (B, T)  1.0 for real completion tokens including the EOS

Left padding the prompts means all completions start at the same column, which
keeps the log-prob slicing a single ``[:, P - 1 : -1]`` for the whole batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tiny_grpo.grpo import gather_logprobs
from tiny_grpo.lora import apply_lora
from tiny_grpo.tasks import chat_messages

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_policy(
    name: str,
    device: torch.device,
    dtype: str = "bfloat16",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_targets: Sequence[str] | None = None,
    gradient_checkpointing: bool = False,
) -> tuple[nn.Module, Any]:
    """Load a Hugging Face causal LM with LoRA adapters, plus its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=_DTYPES[dtype])
    model.to(device)  # type: ignore[arg-type]  # transformers wraps .to() with a narrower signature
    if lora_targets is None:
        apply_lora(model, lora_rank, lora_alpha)
    else:
        apply_lora(model, lora_rank, lora_alpha, lora_targets)
    if gradient_checkpointing:
        # Non-reentrant checkpointing works when only the LoRA weights, not the
        # layer inputs, require grad. It is only active in train() mode.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # No dropout anywhere: the "old" and "current" log-probs must come from the
    # same function or the importance ratio is noise.
    model.eval()
    return model, tokenizer


def eos_token_ids(model: nn.Module, tokenizer: Any) -> list[int]:
    """All tokens that end a turn. Qwen's instruct models stop on either
    ``<|im_end|>`` or ``<|endoftext|>``, and the generation config knows both."""
    ids: set[int] = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for value in (gen_eos, tokenizer.eos_token_id):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, list):
            ids.update(value)
    if not ids:
        raise ValueError("could not determine EOS token ids")
    return sorted(ids)


def encode_prompts(
    tokenizer: Any, questions: Sequence[str], device: torch.device, answer_example: str = "42"
) -> tuple[Tensor, Tensor]:
    """Chat-format and left-pad a batch of questions."""
    texts = [
        tokenizer.apply_chat_template(
            chat_messages(q, answer_example),
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def completion_mask(completion_ids: Tensor, eos_ids: Sequence[int]) -> Tensor:
    """1.0 for every token up to and including the first EOS, 0.0 after it.

    A completion that never emits EOS (hit ``max_new_tokens``) is all ones.
    """
    is_eos = torch.isin(completion_ids, torch.tensor(list(eos_ids), device=completion_ids.device))
    # Number of EOS tokens strictly before each position; the first EOS itself
    # has zero before it, so it stays in the mask.
    eos_before = torch.cumsum(is_eos.long(), dim=-1) - is_eos.long()
    return (eos_before == 0).float()


def is_truncated(completion_ids: Tensor, eos_ids: Sequence[int]) -> Tensor:
    """True for completions that ran out of tokens before emitting any EOS."""
    eos = torch.tensor(list(eos_ids), device=completion_ids.device)
    return ~torch.isin(completion_ids, eos).any(dim=-1)


def position_ids_from_mask(attention_mask: Tensor) -> Tensor:
    """Positions that skip left padding, the same way ``generate`` computes them."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def _chunk_logprobs(hidden: Tensor, weight: Tensor, tokens: Tensor, temperature: float) -> Tensor:
    # Always project in fp32. bf16 logits of size ~25 are only resolved to
    # ~0.1, which is enough noise to show up in the importance ratio. For an
    # fp32 model .float() is a no-op.
    return gather_logprobs(F.linear(hidden.float(), weight.float()), tokens, temperature)


def sequence_logprobs(
    model: nn.Module,
    prompt_ids: Tensor,
    prompt_mask: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
    temperature: float = 1.0,
    chunk_size: int = 64,
) -> Tensor:
    """Per-token ``log pi(completion_t | prompt, completion_<t)``, shape ``(B, T)``.

    Positions where ``completion_mask`` is zero get whatever the model says; the
    caller must multiply by the mask.

    The vocabulary projection is the memory problem here, not the transformer.
    Qwen2.5 has a 151,936-token vocabulary, so logits for 8 completions of 256
    tokens are 8 * 256 * 152k floats: 1.2 GB in fp32 before log-softmax makes
    another copy, and autograd keeps them for the backward pass. So this runs
    the decoder once, keeps only the hidden states at completion positions, and
    projects them to the vocabulary ``chunk_size`` positions at a time. When
    gradients are needed each chunk is checkpointed, which means the backward
    pass recomputes that chunk's logits instead of storing them.
    """
    t = completion_ids.shape[1]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask.to(prompt_mask.dtype)], dim=1)
    decoder = model.get_decoder()  # type: ignore[operator]
    hidden = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
    ).last_hidden_state
    # hidden[:, i] predicts the token at i + 1, so completion token j (at column
    # P + j) is predicted from column P + j - 1.
    hidden = hidden[:, -t - 1 : -1]
    head = model.get_output_embeddings()  # type: ignore[operator]
    if getattr(head, "bias", None) is not None:
        raise NotImplementedError("output projections with a bias are not supported")
    weight = head.weight

    chunks = []
    for start in range(0, t, chunk_size):
        h = hidden[:, start : start + chunk_size]
        tok = completion_ids[:, start : start + chunk_size]
        if torch.is_grad_enabled() and h.requires_grad:
            lp = checkpoint(_chunk_logprobs, h, weight, tok, temperature, use_reentrant=False)
        else:
            lp = _chunk_logprobs(h, weight, tok, temperature)
        chunks.append(lp)
    return torch.cat(chunks, dim=1)


@dataclass
class SamplingParams:
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    greedy: bool = False


def sampling_config(params: SamplingParams, pad_id: int, eos_ids: Sequence[int]) -> Any:
    """A ``GenerationConfig`` that samples from exactly the policy's distribution.

    Every sampling knob is set explicitly. Qwen2.5-0.5B-Instruct ships a
    generation config with ``temperature=0.7, top_p=0.8, top_k=20,
    repetition_penalty=1.1``. If those leak in, the samples come from a
    different distribution than the one whose log-probs we compute, and the
    importance ratio is wrong from the very first step.
    """
    common: dict[str, Any] = {
        "max_new_tokens": params.max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": list(eos_ids),
        "use_cache": True,
        "repetition_penalty": 1.0,
    }
    if params.greedy:
        return GenerationConfig(do_sample=False, **common)
    return GenerationConfig(
        do_sample=True, temperature=params.temperature, top_p=params.top_p, top_k=0, **common
    )


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Tensor,
    prompt_mask: Tensor,
    params: SamplingParams,
    pad_id: int,
    eos_ids: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Sample completions with the KV-cached ``model.generate``.

    Returns ``(completion_ids, completion_mask)``, with tokens after the first EOS
    replaced by ``pad_id``.
    """
    gen_cfg = sampling_config(params, pad_id, eos_ids)
    out = model.generate(
        input_ids=prompt_ids, attention_mask=prompt_mask, generation_config=gen_cfg
    )  # type: ignore[operator]
    completion_ids = out[:, prompt_ids.shape[1] :]
    mask = completion_mask(completion_ids, eos_ids)
    completion_ids = torch.where(
        mask.bool(), completion_ids, torch.full_like(completion_ids, pad_id)
    )
    return completion_ids, mask


def decode_completions(tokenizer: Any, completion_ids: Tensor, mask: Tensor) -> list[str]:
    texts: list[str] = []
    for ids, m in zip(completion_ids.tolist(), mask.tolist(), strict=True):
        kept = [tok for tok, keep in zip(ids, m, strict=True) if keep]
        texts.append(tokenizer.decode(kept, skip_special_tokens=True))
    return texts

"""Log-prob gathering, masking and generation, checked on a tiny random model."""

from typing import Any

import torch
from torch import nn

from tests.conftest import EOS, PAD, VOCAB, build_tiny_model
from tiny_grpo.grpo import LossConfig
from tiny_grpo.lora import adapters_disabled, apply_lora, lora_parameters
from tiny_grpo.policy import (
    SamplingParams,
    completion_mask,
    generate,
    is_truncated,
    position_ids_from_mask,
    sampling_config,
    sequence_logprobs,
)
from tiny_grpo.update import Rollouts, score_rollouts


def _left_pad(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), PAD)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(s)
        mask[i, width - len(s) :] = 1
    return ids, mask


def _naive_logprobs(model: nn.Module, prompt: list[int], completion: list[int]) -> list[float]:
    """One unpadded forward pass per completion token. Slow and obviously right."""
    out = []
    for t in range(len(completion)):
        ctx = torch.tensor([prompt + completion[:t]])
        logits = model(input_ids=ctx).logits[0, -1]
        out.append(float(logits.log_softmax(-1)[completion[t]]))
    return out


# ---------------------------------------------------------------------------
# completion_mask
# ---------------------------------------------------------------------------


def test_completion_mask_includes_first_eos_only() -> None:
    ids = torch.tensor(
        [
            [5, 6, EOS, 7, EOS],  # stop at first EOS, inclusive
            [EOS, 5, 6, 7, 8],  # immediate EOS
            [5, 6, 7, 8, 9],  # truncated, no EOS
        ]
    )
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(completion_mask(ids, [EOS]), expected)
    assert is_truncated(ids, [EOS]).tolist() == [False, False, True]


def test_completion_mask_multiple_eos_ids() -> None:
    ids = torch.tensor([[5, 3, 6, 1]])
    assert completion_mask(ids, [1, 3]).tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_position_ids_skip_left_padding() -> None:
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert position_ids_from_mask(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


# ---------------------------------------------------------------------------
# sequence_logprobs
# ---------------------------------------------------------------------------


def test_logprobs_match_naive_loop_with_padding(tiny_model: nn.Module) -> None:
    prompts = [[5, 9, 12], [7, 8, 20, 21, 22, 23], [30]]
    completions = [[4, 6, EOS], [10, 11, 12, 13, EOS], [14, 15, 16, 17, 18]]
    prompt_ids, prompt_mask = _left_pad(prompts)
    t = max(len(c) for c in completions)
    completion_ids = torch.full((3, t), PAD)
    for i, c in enumerate(completions):
        completion_ids[i, : len(c)] = torch.tensor(c)
    mask = completion_mask(completion_ids, [EOS])

    with torch.no_grad():
        batched = sequence_logprobs(tiny_model, prompt_ids, prompt_mask, completion_ids, mask)
        for i, (p, c) in enumerate(zip(prompts, completions, strict=True)):
            naive = torch.tensor(_naive_logprobs(tiny_model, p, c))
            torch.testing.assert_close(batched[i, : len(c)], naive, atol=1e-5, rtol=1e-4)


def test_chunked_projection_matches_unchunked_values_and_grads(tiny_model: nn.Module) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    with torch.no_grad():
        for p in lora_parameters(tiny_model):
            p.add_(torch.randn_like(p) * 0.1)
    prompt_ids, prompt_mask = _left_pad([[5, 9, 12], [7, 8]])
    completion_ids = torch.randint(2, VOCAB, (2, 7))
    mask = torch.ones(2, 7)
    weights = torch.randn(2, 7)

    results = []
    for chunk in (64, 3, 1):
        tiny_model.zero_grad()
        lp = sequence_logprobs(
            tiny_model, prompt_ids, prompt_mask, completion_ids, mask, chunk_size=chunk
        )
        (lp * weights).sum().backward()
        grads = torch.cat(
            [p.grad.flatten() for p in lora_parameters(tiny_model) if p.grad is not None]
        )
        results.append((lp.detach(), grads))
    for lp, grads in results[1:]:
        torch.testing.assert_close(lp, results[0][0])
        torch.testing.assert_close(grads, results[0][1])


def test_prompt_and_padding_tokens_do_not_affect_masked_loss(tiny_model: nn.Module) -> None:
    """Changing a padding token, or the token after EOS, leaves every masked
    log-prob unchanged, and gradients only flow from completion tokens."""
    prompt_ids, prompt_mask = _left_pad([[5, 9], [7, 8, 20, 21]])
    completion_ids = torch.tensor([[4, EOS, PAD, PAD], [10, 11, 12, EOS]])
    mask = completion_mask(completion_ids, [EOS])

    with torch.no_grad():
        base = sequence_logprobs(tiny_model, prompt_ids, prompt_mask, completion_ids, mask)
        noisy_prompt = prompt_ids.clone()
        noisy_prompt[prompt_mask == 0] = 33  # garbage under the prompt padding
        noisy_completion = completion_ids.clone()
        noisy_completion[mask == 0] = 40  # garbage after EOS
        other = sequence_logprobs(tiny_model, noisy_prompt, prompt_mask, noisy_completion, mask)
    torch.testing.assert_close(base * mask, other * mask, atol=1e-5, rtol=1e-4)


def test_left_padding_matches_unpadded(tiny_model: nn.Module) -> None:
    prompt, completion = [5, 9, 12], [4, 6, 7, EOS]
    ids, pmask = _left_pad([prompt])
    comp = torch.tensor([completion])
    cmask = torch.ones(1, 4)
    with torch.no_grad():
        alone = sequence_logprobs(tiny_model, ids, pmask, comp, cmask)
        padded_ids = torch.cat([torch.full((1, 5), PAD), ids], dim=1)
        padded_mask = torch.cat([torch.zeros(1, 5, dtype=torch.long), pmask], dim=1)
        padded = sequence_logprobs(tiny_model, padded_ids, padded_mask, comp, cmask)
    torch.testing.assert_close(alone, padded, atol=1e-5, rtol=1e-4)


def test_logprobs_match_generate_scores(tiny_model: nn.Module) -> None:
    """The log-probs we train on are the log-probs the sampler actually used."""
    prompt_ids, prompt_mask = _left_pad([[5, 9, 12], [7, 8, 20, 21, 22]])
    torch.manual_seed(0)
    out = tiny_model.generate(  # type: ignore[operator]
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        do_sample=True,
        max_new_tokens=6,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        pad_token_id=PAD,
        eos_token_id=EOS,
        return_dict_in_generate=True,
        output_logits=True,
    )
    completion_ids = out.sequences[:, prompt_ids.shape[1] :]
    mask = completion_mask(completion_ids, [EOS])
    sampler = torch.stack(out.logits, dim=1).log_softmax(-1)
    sampler = sampler.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ours = sequence_logprobs(tiny_model, prompt_ids, prompt_mask, completion_ids, mask)
    torch.testing.assert_close(ours * mask, sampler * mask, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_pads_after_eos(tiny_model: nn.Module) -> None:
    prompt_ids, prompt_mask = _left_pad([[5, 9], [7, 8, 20]])
    torch.manual_seed(0)
    ids, mask = generate(
        tiny_model, prompt_ids, prompt_mask, SamplingParams(max_new_tokens=12), PAD, [EOS]
    )
    assert ids.shape == mask.shape
    assert ids.shape[0] == 2
    assert ids.shape[1] <= 12
    # Every position outside the mask is padding.
    assert (ids[mask == 0] == PAD).all()


def test_sampler_ignores_shipped_generation_defaults() -> None:
    """The checkpoint's own generation config must not reshape the sampling
    distribution. Mimic Qwen's shipped defaults and check that the scores the
    sampler actually drew from are the raw logits (no temperature, top-k,
    top-p or repetition penalty applied)."""
    model: Any = build_tiny_model(vocab=200)
    model.generation_config.update(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1)
    prompt_ids, prompt_mask = _left_pad([[5, 9, 12, 5, 9], [7, 8, 20]])
    cfg = sampling_config(SamplingParams(max_new_tokens=6), PAD, [EOS])
    cfg.update(return_dict_in_generate=True, output_scores=True, output_logits=True)
    out = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask, generation_config=cfg)
    scores, logits = torch.stack(out.scores), torch.stack(out.logits)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, logits)


def test_greedy_generation_is_deterministic(tiny_model: nn.Module) -> None:
    prompt_ids, prompt_mask = _left_pad([[5, 9], [7, 8, 20]])
    params = SamplingParams(max_new_tokens=8, greedy=True)
    a, _ = generate(tiny_model, prompt_ids, prompt_mask, params, PAD, [EOS])
    b, _ = generate(tiny_model, prompt_ids, prompt_mask, params, PAD, [EOS])
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# LoRA and the reference policy
# ---------------------------------------------------------------------------


def test_lora_starts_as_identity_and_disables_cleanly(tiny_model: nn.Module) -> None:
    x = torch.randint(2, VOCAB, (2, 7))
    with torch.no_grad():
        before = tiny_model(input_ids=x).logits
    modules = apply_lora(tiny_model, rank=4, alpha=8.0, targets=["q_proj", "v_proj", "down_proj"])
    assert len(modules) == 2 * 3
    assert all(p.requires_grad for p in lora_parameters(tiny_model))
    assert sum(p.requires_grad for p in tiny_model.parameters()) == 2 * len(modules)

    with torch.no_grad():
        # B = 0 at init, so the adapted model is the base model.
        torch.testing.assert_close(tiny_model(input_ids=x).logits, before)
        for p in lora_parameters(tiny_model):
            p.add_(torch.randn_like(p) * 0.5)
        moved = tiny_model(input_ids=x).logits
        with adapters_disabled(tiny_model):
            ref = tiny_model(input_ids=x).logits
        again = tiny_model(input_ids=x).logits
    assert not torch.allclose(moved, before)
    torch.testing.assert_close(ref, before)  # reference == frozen base weights
    torch.testing.assert_close(again, moved)  # adapters come back on


def test_score_rollouts_reference_uses_base_weights(tiny_model: nn.Module) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    prompt_ids, prompt_mask = _left_pad([[5, 9], [7, 8, 20]])
    completion_ids = torch.tensor([[4, 6, EOS], [10, 11, 12]])
    mask = completion_mask(completion_ids, [EOS])
    r = Rollouts(prompt_ids, prompt_mask, completion_ids, mask, torch.zeros(2))

    ref0 = score_rollouts(tiny_model, r, micro_batch_size=1, reference=True)
    cur0 = score_rollouts(tiny_model, r, micro_batch_size=2)
    torch.testing.assert_close(ref0, cur0)
    with torch.no_grad():
        for p in lora_parameters(tiny_model):
            p.add_(torch.randn_like(p) * 0.5)
    ref1 = score_rollouts(tiny_model, r, micro_batch_size=1, reference=True)
    cur1 = score_rollouts(tiny_model, r, micro_batch_size=2)
    torch.testing.assert_close(ref1, ref0)
    assert not torch.allclose(cur1 * mask, cur0 * mask)


def test_loss_config_defaults_are_the_paper_values() -> None:
    cfg = LossConfig()
    assert (cfg.clip_low, cfg.clip_high, cfg.kl_coef) == (0.2, 0.2, 0.04)

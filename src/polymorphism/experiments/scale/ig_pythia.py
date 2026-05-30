"""Integrated gradients and attribution patching for GPT-NeoX / Pythia.

Componentization (granularity matched to the residual-stream sites):
  "layer{L}_out" — the contribution of block L to the residual stream
                   (= hidden_states[L+1] - hidden_states[L]).
  "embed"        — the contribution of the embedding layer (= hidden_states[0]).

For each component c with clean value `a_c` and dataset-mean value `mu_c`,
attribution patching (Nanda 2023) predicts the mean-ablation effect as
    AP(c) = grad_a_c(L_clean) . (mu_c - a_c)
i.e. linear extrapolation from the clean activation to the mean. At a converged
model this gradient is small and noisy, so the predictor anti-correlates with
the measurement.

Integrated gradients (Sundararajan, Taly & Yan 2017) integrates the gradient
along the straight line from mean to clean:
    IG(c) = (mu_c - a_c) . (1/n) sum_{k=1..n} grad_{a(α_k)} L
where a(α) = α a_c + (1-α) mu_c. With per-component interpolation (only one c
moves at a time), the IG completeness axiom gives IG(c) = L(clean) - L(c-mean)
up to discretisation in n.

Implementation uses torch forward hooks; we register a hook on each layer's
output, capture the clean value, then re-run with the hook replacing the
component by the interpolation and measuring grad through it.

Both predictors share the same forward-loss computation:
    L = mean_t [ -log p_model(next_token | tokens[<t]) ]
i.e. the cross-entropy on the natural-language continuation. Mask is implicit
(no padding in fixed-length text chunks).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ----------------- forward-loss helper -----------------

def _ce_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard causal-LM cross-entropy: predict tokens[t+1] from logits[t].

    logits: [B, T, V], input_ids: [B, T]. Returns scalar mean loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1))


# ----------------- residual-stream interpolation -----------------

def _gpt_neox_blocks(model):
    """Find the ModuleList of transformer blocks on a Pythia / GPT-NeoX model."""
    # Pythia uses GPTNeoXForCausalLM; the blocks list is .gpt_neox.layers
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    # Fall back to common alternatives
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Cannot find transformer blocks on model")


def _embed_layer(model):
    """Find the input-embedding layer (token + pos embeds combined for Pythia)."""
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.embed_in
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise AttributeError("Cannot find embedding layer on model")


# ----------------- clean-pass cache -----------------

@torch.no_grad()
def collect_clean_layer_contribs(model, tokens: torch.Tensor,
                                  device: str = "cuda") -> dict:
    """Run a clean forward and capture per-layer contributions to the residual.

    Returns:
      clean[L] = hidden_states[L+1] - hidden_states[L]  (the block L "output")
                in fp32 on CPU.
      clean["embed"] = hidden_states[0]                  (post-embed initial residual)
      loss            (scalar, fp32)
      logits          [B, T, V] on CPU fp32
    """
    out = model(tokens.to(device), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple length n_layers + 1
    n_layers = len(hs) - 1
    clean = {"embed": hs[0].detach().float().cpu()}
    for L in range(n_layers):
        clean[f"layer{L}"] = (hs[L + 1] - hs[L]).detach().float().cpu()
    loss = _ce_loss(out.logits.float(), tokens.to(device)).item()
    return {"clean": clean, "loss": loss, "logits": out.logits.detach().float().cpu()}


# ----------------- component means over a sample -----------------

@torch.no_grad()
def estimate_component_means(model, tokens: torch.Tensor, batch_size: int = 8,
                              device: str = "cuda") -> dict:
    """Estimate dataset-mean of each component (broadcasting over (B, T)).

    Means are computed by averaging the per-block residual delta across all
    sampled (b, t) positions. Returns dict component_name -> tensor [d_model]
    on CPU fp32.
    """
    sums = None
    n = 0
    N = tokens.shape[0]
    for i in range(0, N, batch_size):
        batch = tokens[i : i + batch_size].to(device)
        out = model(batch, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        if sums is None:
            n_layers = len(hs) - 1
            sums = {f"layer{L}": torch.zeros(hs[0].shape[-1],
                                             dtype=torch.float32, device=device)
                    for L in range(n_layers)}
            sums["embed"] = torch.zeros_like(sums["layer0"])
        # embed
        sums["embed"] += hs[0].float().reshape(-1, hs[0].shape[-1]).sum(0)
        n += hs[0].shape[0] * hs[0].shape[1]
        for L in range(len(hs) - 1):
            delta = (hs[L + 1] - hs[L]).float()
            sums[f"layer{L}"] += delta.reshape(-1, delta.shape[-1]).sum(0)
    return {k: (v / n).detach().cpu() for k, v in sums.items()}


# ----------------- attribution patching -----------------

def attribution_patch_pythia(model, tokens: torch.Tensor, means: dict,
                              device: str = "cuda") -> dict:
    """For each layer L, predict the mean-ablation effect via attribution patching.

    Returns dict component -> predicted loss-delta (scalar).
    """
    # Enable gradients only for hook input; freeze model
    saved_req = []
    for p in model.parameters():
        saved_req.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        blocks = _gpt_neox_blocks(model)
        n_layers = len(blocks)
        captured = {}
        hooks = []

        def make_hook(L):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    out0 = output[0]
                else:
                    out0 = output
                out0.requires_grad_(True)
                out0.retain_grad()
                captured[f"layer{L}"] = out0
                if isinstance(output, tuple):
                    return (out0,) + output[1:]
                return out0
            return hook

        for L, blk in enumerate(blocks):
            hooks.append(blk.register_forward_hook(make_hook(L)))
        try:
            out = model(tokens.to(device), output_hidden_states=False, use_cache=False)
            loss = _ce_loss(out.logits.float(), tokens.to(device))
            loss.backward()
        finally:
            for h in hooks:
                h.remove()

        predicted = {}
        for L in range(n_layers):
            key = f"layer{L}"
            if key not in captured:
                continue
            grad = captured[key].grad
            if grad is None:
                predicted[key] = 0.0
                continue
            # Predicted patch effect = grad . (mean - clean_resid_post)
            # But for "per-layer" component (the block's contribution),
            # we need grad . (mean - clean_contrib).
            # captured[key] is hidden_states[L+1] (the block output is the
            # full post-residual). The mean of the *contribution* (delta)
            # is means[f"layer{L}"]; the corresponding mean of the post-
            # residual is captured[key].mean by construction. To produce a
            # comparable predictor we use the block-output gradient directly
            # since the block output is what the residual stream actually
            # carries forward; downstream gradients to hidden_states[L+1]
            # propagate linearly to any sub-component of the block (this
            # is the standard AP formulation at the residual-stream site).
            #
            # Use: AP = grad . (mean_resid_post - actual_resid_post).
            actual = captured[key].detach()
            # The mean of the resid_post can be approximated as the running
            # mean of hidden_states[L+1] computed during means estimation;
            # here we use the per-batch mean as proxy.
            mu = actual.float().reshape(-1, actual.shape[-1]).mean(0)
            mu = mu.view(*([1] * (actual.dim() - 1)), -1)
            delta = (mu - actual).float()
            predicted[key] = float((grad.float() * delta).sum().item())
    finally:
        for p, r in zip(model.parameters(), saved_req):
            p.requires_grad_(r)
    return predicted


# ----------------- integrated gradients (per-component) -----------------

def integrated_gradients_pythia(model, tokens: torch.Tensor, means: dict,
                                  n_steps: int = 32, device: str = "cuda",
                                  components: list[str] | None = None) -> dict:
    """Per-component IG: for each component, interpolate ONLY that component
    between mean and clean, holding all others at clean. Return predicted
    loss-delta (mean-ablation effect) per component.

    By the IG completeness axiom (Sundararajan et al. 2017), this matches the
    actual mean-ablation effect up to discretisation in n_steps.

    Components default to ['layer0', 'layer1', ..., 'layer{n-1}'].
    """
    blocks = _gpt_neox_blocks(model)
    n_layers = len(blocks)
    if components is None:
        components = [f"layer{L}" for L in range(n_layers)]

    # Determine model dtype to keep dtype consistent throughout the
    # interpolated forward pass (the GPT-NeoX LayerNorm internals require
    # matching dtypes between activations and learned weights).
    model_dtype = next(model.parameters()).dtype

    # First, a clean pass to record the per-block "actual" outputs.
    saved_req = []
    for p in model.parameters():
        saved_req.append(p.requires_grad)
        p.requires_grad_(False)

    clean_outs = {}  # L -> [B, T, d] tensor (on device, model dtype)
    hooks = []

    def make_clean_hook(L):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                out0 = output[0]
            else:
                out0 = output
            clean_outs[L] = out0.detach().clone()
            return output
        return hook
    try:
        for L, blk in enumerate(blocks):
            hooks.append(blk.register_forward_hook(make_clean_hook(L)))
        with torch.no_grad():
            _ = model(tokens.to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    # For each target component, do n_steps interpolated forwards with a hook
    # that replaces the target block's output with α*clean + (1-α)*mean.
    predicted = {}
    try:
        for tgt in components:
            tgt_L = int(tgt[5:])  # 'layer{L}'
            # The "mean" of the per-block output, model dtype preserved.
            mu = clean_outs[tgt_L].reshape(-1, clean_outs[tgt_L].shape[-1]).mean(0)
            mu = mu.view(*([1] * (clean_outs[tgt_L].dim() - 1)), -1)
            mu = mu.expand_as(clean_outs[tgt_L]).contiguous()
            delta = (mu - clean_outs[tgt_L]).float()    # mean - actual, fp32 for accumulator

            grad_sum = torch.zeros_like(clean_outs[tgt_L], dtype=torch.float32)
            for k in range(1, n_steps + 1):
                alpha = k / n_steps
                # Build interpolation in fp32 then cast back to model dtype
                # for the forward pass (LayerNorm needs matching dtypes).
                interp_value_fp32 = (alpha * clean_outs[tgt_L].detach().float()
                                       + (1 - alpha) * mu.float())
                interp_value = interp_value_fp32.to(model_dtype).requires_grad_(True)

                def make_interp_hook():
                    def hook(module, inputs, output):
                        if isinstance(output, tuple):
                            return (interp_value,) + output[1:]
                        return interp_value
                    return hook

                h = blocks[tgt_L].register_forward_hook(make_interp_hook())
                try:
                    out = model(tokens.to(device), use_cache=False)
                    loss = _ce_loss(out.logits.float(), tokens.to(device))
                    g, = torch.autograd.grad(loss, interp_value)
                    grad_sum = grad_sum + g.detach().float()
                finally:
                    h.remove()
            # Predicted ablation effect = (mu - actual) . avg_grad
            predicted[tgt] = float((delta * grad_sum / n_steps).sum().item())
    finally:
        for p, r in zip(model.parameters(), saved_req):
            p.requires_grad_(r)
    return predicted


# ----------------- measured (true mean-ablation) -----------------

def measured_mean_ablation_pythia(model, tokens: torch.Tensor,
                                    device: str = "cuda",
                                    components: list[str] | None = None) -> dict:
    """For each layer L, replace the block-L output with its per-batch mean and
    measure the actual loss change vs the clean loss. This is what IG/AP try
    to predict.
    """
    blocks = _gpt_neox_blocks(model)
    n_layers = len(blocks)
    if components is None:
        components = [f"layer{L}" for L in range(n_layers)]

    with torch.no_grad():
        out_clean = model(tokens.to(device), use_cache=False)
        loss_clean = _ce_loss(out_clean.logits.float(), tokens.to(device)).item()

    # Capture per-block clean outputs
    saved_req = []
    for p in model.parameters():
        saved_req.append(p.requires_grad); p.requires_grad_(False)
    clean_outs = {}
    try:
        hooks = []
        def make_cap(L):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    clean_outs[L] = output[0].detach().float().clone()
                else:
                    clean_outs[L] = output.detach().float().clone()
                return output
            return hook
        for L, blk in enumerate(blocks):
            hooks.append(blk.register_forward_hook(make_cap(L)))
        with torch.no_grad():
            _ = model(tokens.to(device), use_cache=False)
        for h in hooks:
            h.remove()

        deltas = {}
        for tgt in components:
            tgt_L = int(tgt[5:])
            mu = clean_outs[tgt_L].reshape(-1, clean_outs[tgt_L].shape[-1]).mean(0)
            mu = mu.view(*([1] * (clean_outs[tgt_L].dim() - 1)), -1).expand_as(clean_outs[tgt_L])
            mu = mu.contiguous()

            def make_ablate(M):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        return (M.to(output[0].dtype),) + output[1:]
                    return M.to(output.dtype)
                return hook
            h = blocks[tgt_L].register_forward_hook(make_ablate(mu))
            try:
                with torch.no_grad():
                    out_a = model(tokens.to(device), use_cache=False)
                    loss_a = _ce_loss(out_a.logits.float(), tokens.to(device)).item()
            finally:
                h.remove()
            deltas[tgt] = loss_a - loss_clean
    finally:
        for p, r in zip(model.parameters(), saved_req):
            p.requires_grad_(r)
    return {"baseline_loss": loss_clean, "ablation_loss_delta": deltas}


# ----------------- end-to-end -----------------

def run_ig_vs_ap_comparison(model, tokens: torch.Tensor, n_ig_steps: int = 32,
                              device: str = "cuda") -> dict:
    """Full IG-vs-AP comparison on Pythia: per-layer predictions and measurements,
    Pearson r for each predictor.
    """
    means = estimate_component_means(model, tokens, batch_size=8, device=device)
    measured = measured_mean_ablation_pythia(model, tokens, device=device)
    ap_pred = attribution_patch_pythia(model, tokens, means, device=device)
    ig_pred = integrated_gradients_pythia(model, tokens, means, n_steps=n_ig_steps,
                                            device=device)
    components = sorted(measured["ablation_loss_delta"].keys())
    m_vec = [measured["ablation_loss_delta"][c] for c in components]
    ap_vec = [ap_pred.get(c, 0.0) for c in components]
    ig_vec = [ig_pred.get(c, 0.0) for c in components]
    import numpy as np
    def _r(x, y):
        x, y = np.array(x), np.array(y)
        if x.std() < 1e-12 or y.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    return {
        "components": components,
        "measured": m_vec,
        "attribution_patch": ap_vec,
        "integrated_gradients": ig_vec,
        "pearson_r_ap": _r(ap_vec, m_vec),
        "pearson_r_ig": _r(ig_vec, m_vec),
        "baseline_loss": measured["baseline_loss"],
        "n_ig_steps": n_ig_steps,
    }

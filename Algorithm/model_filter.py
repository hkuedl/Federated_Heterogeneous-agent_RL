from copy import deepcopy
import numpy as np
import torch


def _flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in state_dict.values()])


def _unflatten_state_dict(vector: torch.Tensor, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    ptr = 0
    for k, param in template.items():
        n = param.numel()
        out[k] = vector[ptr : ptr + n].view_as(param)
        ptr += n
    return out


def _clip_updates_by_norm(updates: list[dict[str, torch.Tensor]]) -> tuple[list[dict[str, torch.Tensor]], float]:
    flat_updates = []
    valid_updates = []
    for update in updates:
        flat = _flatten_state_dict(update)
        if torch.isnan(flat).any():
            continue
        flat_updates.append(flat)
        valid_updates.append(update)

    if not flat_updates:
        return [], 0.0

    flat_stack = torch.stack(flat_updates, dim=0)

    norms = torch.norm(flat_stack, dim=1, keepdim=True)
    clip_value = float(torch.median(norms))

    scale = torch.where(norms > 0, torch.clamp(clip_value / norms, max=1.0), torch.ones_like(norms))
    clipped_flat = flat_stack * scale

    clipped_updates = []
    for i, update in enumerate(valid_updates):
        clipped_updates.append(_unflatten_state_dict(clipped_flat[i], update))

    return clipped_updates, clip_value


def _compute_layer_weights(
    updates: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:

    if not updates:
        return {}

    num_clients = len(updates)
    weights_by_key: dict[str, torch.Tensor] = {}

    for key in updates[0].keys():
        stacked = torch.stack([u[key].flatten() for u in updates], dim=0)

        # 1) Magnitude score
        norms = torch.norm(stacked.float(), dim=1)
        median = norms.median()
        std = norms.std(unbiased=False)
        std = std if std > 0 else 1.0
        l2_scores = torch.abs((norms - median) / std)
        l2_scores = l2_scores / l2_scores.sum()

        # 2) Direction score
        sign = torch.sign(stacked)
        sign_sum = sign.sum(dim=1)
        abs_sign_sum = torch.abs(sign).sum(dim=1)
        sign_score = torch.zeros_like(sign_sum)
        nonzero_mask = abs_sign_sum > 0
        sign_score[nonzero_mask] = 0.5 * (
            1 + sign_sum[nonzero_mask] / abs_sign_sum[nonzero_mask]
        )

        if sign_score.sum() > 0:
            sign_scores = sign_score / sign_score.sum()
        else:
            sign_scores = torch.ones(num_clients, device=sign_score.device) / num_clients

        # Combine magnitude and direction scores
        combined = (l2_scores + sign_scores) / 2.0
        weights_by_key[key] = combined

    return weights_by_key


def Filter(local_updates: list[dict[str, torch.Tensor]], global_model: dict[str, torch.Tensor], args: dict):
    # 1) Clip the updates
    clipped_updates, _clip_value = _clip_updates_by_norm(local_updates)
    if not clipped_updates:
        return global_model, None

    # 2) Compute layer-wise aggregation weights
    weights_by_key = _compute_layer_weights(clipped_updates)

    # 3) Weighted aggregation
    for key, w in weights_by_key.items():
        aggregated = torch.zeros_like(clipped_updates[0][key])
        for i, update in enumerate(clipped_updates):
            aggregated += w[i] * update[key]
        global_model[key].data += aggregated.data

    return global_model, None

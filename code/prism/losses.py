"""Physics-informed losses and augmentations."""

import torch
import torch.nn.functional as F


def monotonicity_loss(
    x_batch: torch.Tensor,
    y_pred: torch.Tensor,
    mono_pairs: list,
) -> torch.Tensor:
    """Soft pairwise monotonicity penalty."""
    batch_size = x_batch.shape[0]
    if batch_size < 2 or not mono_pairs:
        return torch.tensor(0.0, device=x_batch.device)

    total = torch.tensor(0.0, device=x_batch.device)
    for feat_idx, target_idx, direction in mono_pairs:
        feat = x_batch[:, feat_idx]
        pred = y_pred[:, target_idx]
        delta_feat = feat.unsqueeze(0) - feat.unsqueeze(1)
        delta_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
        feat_up = delta_feat * direction > 0
        violation = F.relu(-delta_pred * direction)
        penalty = (violation * feat_up.float()).sum() / feat_up.float().sum().clamp_min(1.0)
        total = total + penalty
    return total / len(mono_pairs)


def perturbation_monotonicity_loss(
    forward_fn,
    x_batch: torch.Tensor,
    y_ref: torch.Tensor,
    mono_pairs: list,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Ceteris-paribus monotonicity penalty using +epsilon feature sweeps."""
    if not mono_pairs:
        return torch.tensor(0.0, device=x_batch.device)

    cache = {}
    total = torch.tensor(0.0, device=x_batch.device)
    for feat_idx, target_idx, direction in mono_pairs:
        if feat_idx not in cache:
            x_plus = x_batch.detach().clone()
            x_plus[:, feat_idx] = x_plus[:, feat_idx] + epsilon
            cache[feat_idx] = forward_fn(x_plus)
        y_plus = cache[feat_idx]
        delta = (y_plus[:, target_idx] - y_ref[:, target_idx].detach()) * direction
        total = total + F.relu(-delta).mean()
    return total / len(mono_pairs)


def large_range_egr_mono_loss(
    forward_fn,
    xb: torch.Tensor,
    y_hat: torch.Tensor,
    egr_mono_pairs: list,
    egr_idx: int = 2,
    epsilons: list | None = None,
) -> torch.Tensor:
    """Enforce EGR monotonicity over multiple perturbation scales."""
    if epsilons is None:
        epsilons = [0.2, 0.5, 1.0]
    if not egr_mono_pairs:
        return torch.tensor(0.0, device=xb.device)

    cache = {}
    total = torch.tensor(0.0, device=xb.device)
    n_pairs = len([p for p in egr_mono_pairs if p[0] == egr_idx])
    if n_pairs == 0:
        return total

    for eps in epsilons:
        if eps not in cache:
            x_pert = xb.detach().clone()
            x_pert[:, egr_idx] = x_pert[:, egr_idx] + eps
            cache[eps] = forward_fn(x_pert)
        y_pert = cache[eps]
        for feat_idx, out_idx, direction in egr_mono_pairs:
            if feat_idx != egr_idx:
                continue
            delta = (y_pert[:, out_idx] - y_hat[:, out_idx].detach()) * direction
            total = total + F.relu(-delta).mean()

    return total / (len(epsilons) * n_pairs)


def egr_axis_augmentation(
    xb: torch.Tensor,
    acpb: torch.Tensor,
    yb: torch.Tensor,
    egr_idx: int = 2,
    aug_scale: float = 0.4,
    aug_prob: float = 0.3,
) -> tuple:
    """Create synthetic high-EGR samples with correlation-based pseudo-label shifts."""
    batch_size = xb.shape[0]
    mask = torch.rand(batch_size, device=xb.device) < aug_prob
    if mask.sum() == 0:
        return xb, acpb, yb

    delta = torch.zeros(batch_size, device=xb.device)
    n_aug = int(mask.sum())
    delta[mask] = torch.rand(n_aug, device=xb.device) * aug_scale + 0.1

    xb_aug = xb.clone()
    xb_aug[:, egr_idx] = xb[:, egr_idx] + delta

    egr_acp_corr = [+0.237, +0.340, +0.352, -0.191]
    egr_y_corr = [-0.294, 0.009, -0.227, -0.213]

    acpb_aug = acpb.clone()
    for acp_idx, corr in enumerate(egr_acp_corr):
        acpb_aug[:, acp_idx] = acpb[:, acp_idx] + corr * delta

    yb_aug = yb.clone()
    for y_idx, corr in enumerate(egr_y_corr):
        yb_aug[:, y_idx] = yb[:, y_idx] + corr * delta

    return xb_aug, acpb_aug, yb_aug


def curvature_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_idx: int = 0,
    feat_idx: int = 7,
) -> torch.Tensor:
    """Penalize second derivative of model output with respect to an input feature."""
    x = x.detach().clone().requires_grad_(True)
    y = model.predict(x)[:, target_idx]

    grad1 = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0][:, feat_idx]
    grad2 = torch.autograd.grad(
        grad1, x, grad_outputs=torch.ones_like(grad1), create_graph=True, retain_graph=True
    )[0][:, feat_idx]

    return torch.mean(grad2**2)


def supervised_physics_contrastive_loss(
    embeddings: torch.Tensor,
    acp_targets: torch.Tensor,
    temperature: float = 0.07,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Supervised contrastive loss weighted by ACP-space similarity."""
    z = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(z, z.t()) / temperature

    dist_acp = torch.cdist(acp_targets, acp_targets)
    mask = torch.exp(-(dist_acp**2) / (2 * sigma**2))

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

    return -mean_log_prob_pos.mean()


def local_neighborhood_mixup(
    x_batch: torch.Tensor,
    acp_batch: torch.Tensor,
    y_batch: torch.Tensor,
    k: int = 5,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Mix each sample with a nearby batch neighbor in normalized input space."""
    batch_size = x_batch.shape[0]
    if batch_size < 2 or k <= 0 or alpha <= 0:
        return None

    with torch.no_grad():
        dist = torch.cdist(x_batch.detach(), x_batch.detach())
        dist.fill_diagonal_(float("inf"))
        k_eff = min(k, batch_size - 1)
        neighbors = torch.topk(dist, k=k_eff, largest=False).indices
        choice = torch.randint(0, k_eff, (batch_size,), device=x_batch.device)
        partner_idx = neighbors[torch.arange(batch_size, device=x_batch.device), choice]
        lam = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1)).to(x_batch.device)
        lam = torch.maximum(lam, 1.0 - lam)

    x_mix = lam * x_batch + (1.0 - lam) * x_batch[partner_idx]
    acp_mix = lam * acp_batch + (1.0 - lam) * acp_batch[partner_idx]
    y_mix = lam * y_batch + (1.0 - lam) * y_batch[partner_idx]
    return x_mix, acp_mix, y_mix

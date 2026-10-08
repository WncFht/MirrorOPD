"""Monte Carlo check of K4 reverse-KL value and gradient."""

import torch

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(7)
    logits = torch.tensor([0.7, -0.2, 1.1, -1.0], requires_grad=True)
    q_logits = torch.tensor([-0.3, 0.4, 0.2, -0.8])
    p = torch.softmax(logits, 0)
    q = torch.softmax(q_logits, 0)
    exact_value = torch.sum(p * (torch.log(p) - torch.log(q)))
    exact_grad = torch.autograd.grad(exact_value, logits, retain_graph=True)[0]
    n = 200_000
    idx = torch.multinomial(p.detach(), n, replacement=True)
    lp = torch.log_softmax(logits, 0)[idx]
    lq = torch.log(q)[idx]
    w = torch.exp(lq - lp)
    ratio = p[idx] / torch.detach(p[idx])
    k4 = ratio * (-torch.log(w))
    estimate = k4.mean()
    grad = torch.autograd.grad(estimate, logits)[0]
    print(f"exact_value={exact_value.item():.8f}")
    print(
        f"mc_value={estimate.item():.8f} abs_err={abs(estimate.item() - exact_value.item()):.3e}"
    )
    print(f"grad_l2_err={torch.linalg.vector_norm(grad - exact_grad).item():.3e}")
    assert abs(estimate.item() - exact_value.item()) < 2.5e-2
    assert torch.linalg.vector_norm(grad - exact_grad).item() < 2.5e-2


if __name__ == "__main__":
    main()

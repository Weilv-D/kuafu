import torch

from rl.train.distributions import TanhGaussian


def test_tanh_gaussian_bounded():
    loc = torch.zeros(10000, 6)
    scale = torch.full((10000, 6), 0.5)
    dist = TanhGaussian(loc, scale)
    actions = dist.sample()
    assert actions.shape == (10000, 6)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_log_prob_finite():
    loc = torch.randn(4, 6)
    scale = torch.full((4, 6), 0.3)
    dist = TanhGaussian(loc, scale)
    actions = torch.linspace(-0.99, 0.99, 4 * 6).reshape(4, 6)
    lp = dist.log_prob(actions)
    assert lp.shape == (4, 6)
    assert torch.isfinite(lp).all()


def test_entropy_has_gradient():
    loc = torch.randn(8, 6, requires_grad=True)
    scale = torch.rand(8, 6) + 0.1
    scale = scale.detach().requires_grad_(True)
    dist = TanhGaussian(loc, scale)
    entropy = dist.entropy()
    loss = entropy.sum()
    loss.backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert torch.any(loc.grad.abs() > 0)
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    assert torch.any(scale.grad.abs() > 0)


def test_log_prob_matches_reference():
    torch.manual_seed(0)
    loc = torch.randn(3, 6)
    scale = torch.full((3, 6), 0.4)
    dist = TanhGaussian(loc, scale)
    actions = torch.tanh(torch.randn(3, 6))
    ref = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(loc, scale),
        [torch.distributions.transforms.TanhTransform()],
    )
    lp = dist.log_prob(actions)
    lp_ref = ref.log_prob(actions)
    assert torch.allclose(lp, lp_ref, atol=1e-5)

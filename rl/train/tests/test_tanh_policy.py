import torch

from rl.train.distributions import TanhGaussian
from rl.train.tanh_actor_critic import TanhActorCritic


def test_tanh_gaussian_is_bounded_and_has_finite_log_probability():
    loc = torch.randn(8, 6, requires_grad=True)
    dist = TanhGaussian(loc, torch.full_like(loc, 0.3))
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()
    entropy = dist.entropy()
    assert torch.all(action <= 1.0) and torch.all(action >= -1.0)
    assert torch.isfinite(log_prob)
    assert entropy.shape == loc.shape and torch.isfinite(entropy).all()
    log_prob.backward()
    assert torch.isfinite(loc.grad).all()


def test_inference_uses_same_tanh_transform_as_export():
    policy = TanhActorCritic(4, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    observation = torch.randn(3, 4)
    assert torch.allclose(policy.act_inference(observation), torch.tanh(policy.actor(observation)))

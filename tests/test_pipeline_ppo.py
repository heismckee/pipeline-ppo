import numpy as np

from pipeline_ppo import (MLP, N_ACTIONS, OBS_DIM, PipelineEnv, PPOConfig, behavior_clone, evaluate, gae,
                          playbook_action, summarize, train)


def test_env_shapes_and_episode_end():
    env = PipelineEnv(32, seed=0)
    obs = env.reset()
    assert obs.shape == (32, OBS_DIM)
    for _ in range(40):
        obs, r, d, info = env.step(playbook_action(obs))
        assert r.shape == (32,) and d.shape == (32,)
    assert (env.week < 26).all()          # finished deals are replaced, so no slot outlives the 26-week cap


def test_backprop_matches_numerical_gradient():
    m = MLP(6, 3, hidden=8, seed=1, out_scale=1.0)
    x = np.random.default_rng(2).normal(size=(5, 6))
    out, cache = m.forward(x)
    g = m.backward(cache, np.ones_like(out))
    for key, idx in [("W1", (2, 3)), ("W2", (0, 7)), ("W3", (4, 1)), ("b1", (5,))]:
        eps = 1e-6
        m.p[key][idx] += eps; up = m(x).sum()
        m.p[key][idx] -= 2 * eps; down = m(x).sum()
        m.p[key][idx] += eps
        assert abs(g[key][idx] - (up - down) / (2 * eps)) < 1e-5


def test_gae_reduces_to_discounted_sum_when_lambda_is_one():
    rew = np.array([[0.0], [0.0], [1.0]])
    val = np.zeros((3, 1)); done = np.array([[0.0], [0.0], [1.0]])
    _, ret = gae(rew, val, done, np.zeros(1), gamma=0.9, lam=1.0)
    assert np.allclose(ret.ravel(), [0.81, 0.9, 1.0])


def test_value_loss_training_reduces_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 4)); y = (x[:, :1] * 0.5 - x[:, 1:2] * 0.3)
    critic = MLP(4, 1, hidden=16, seed=0, out_scale=1.0)
    before = 0.5 * ((critic(x) - y) ** 2).mean()
    for _ in range(300):
        v, cache = critic.forward(x)
        critic.adam(critic.backward(cache, (v - y) / len(x)), lr=3e-3, clip=10)
    after = 0.5 * ((critic(x) - y) ** 2).mean()
    assert after < before * 0.2


def test_behavior_cloning_copies_the_playbook():
    actor = MLP(OBS_DIM, N_ACTIONS, 64, seed=0, out_scale=0.01)
    assert behavior_clone(actor, steps=200) > 0.95


def test_short_ppo_run_beats_the_playbook():
    actor, critic, ref, hist, _ = train(PPOConfig(iterations=60, lr_actor=1e-3, seed=3), verbose=False, eval_every=60)
    ppo = summarize(evaluate(lambda o: actor(o).argmax(1), 3000, seed=7))["net_per_deal"]
    base = summarize(evaluate(playbook_action, 3000, seed=7))["net_per_deal"]
    assert ppo > base

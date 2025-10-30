import numpy as np 


class PredictEnv:
    def __init__(self, model, env_name):
        self.model = model  # EnsembleDynamicsModel
        self.env_name = env_name
        self.max_steps = 1e3

    def _termination_fn(self, env_name, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        if 'hopper' in env_name:
            if 'no' in env_name:
                return np.zeros((next_obs.shape[0], 1), dtype=bool)
            else:
                states = next_obs[:, 2:]
                height = next_obs[:, 1]
                angle = next_obs[:, 2]
                not_terminated = np.isfinite(next_obs).all(axis=-1) \
                                 * (np.abs(states) < 100).all(axis=-1) \
                                 * (height > .7) \
                                 * (np.abs(angle) < .2)
                terminated = ~not_terminated
                terminated = terminated[:, None]
                return terminated
        elif "walker2d" in env_name:
            if 'no' in env_name:
                return np.zeros((next_obs.shape[0], 1), dtype=bool)
            else:
                height = next_obs[:, 1]
                angle = next_obs[:, 2]
                not_terminated = (height > 0.8) \
                           * (height < 2.0) \
                           * (angle > -1.0) \
                           * (angle < 1.0)
                terminated = ~not_terminated
                terminated = terminated[:, None]
                return terminated
        elif 'swimmer' in env_name or 'halfcheetah' in env_name:
            # Never terminate
            return np.zeros((next_obs.shape[0], 1), dtype=bool)
        else:
            raise NotImplementedError(f"_termination_fn not implemented for env: {self.env_name}")

    def _get_logprob(self, x, means, variances):
        # Aleatoric + epistemic fit, evaluate how likely the sample is under the whole ensemble mixture
        k = x.shape[-1]
        log_prob = -1 / 2 * (
                k * np.log(2 * np.pi)
                + np.log(variances).sum(-1)
                + (np.power(x - means, 2) / variances).sum(-1)
        )  # (B, bs)
        # Ensemble as equal-weight Gaussian mixture
        prob = np.exp(log_prob).sum(0)  # (bs, )
        log_prob = np.log(prob)  # miss -log(ensemble_size) (just a constant offset)

        # Epistemic disagreement
        stds = np.std(means, 0).mean(-1)  # (bs, )

        # Larger log_prob and smaller stds indicate better model fit
        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)                     # (bs, s_dim + a_dim)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)  # (B, bs, r_dim + s_dim)
        ensemble_model_means[:, :, 1:] += obs                                   # s' = s + delta_s
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means     # (B, bs, r_dim + s_dim)
        else:
            ensemble_samples = ensemble_model_means + \
                               np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        # Randomly choose one elite model for each sample in the batch
        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)  # (bs, )
        batch_idxes = np.arange(0, batch_size)                                         # (bs, )
        samples = ensemble_samples[model_idxes, batch_idxes]                # (bs, r_dim + s_dim)
        model_means = ensemble_model_means[model_idxes, batch_idxes]        # (bs, r_dim + s_dim)
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]          # (bs, r_dim + s_dim)

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)  # (bs, 1)

        batch_size = model_means.shape[0]
        return_means = np.concatenate(
            (model_means[:, :1], terminals, model_means[:, 1:]), axis=-1
        )  # (bs, r_dim + 1 + s_dim)
        return_stds = np.concatenate(
            (model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1
        )

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {
            'mean': return_means,  # per-dim mean (r, terminal-as-0/1, s)
            'std': return_stds,    # per-dim std (0 for terminal)
            'log_prob': log_prob,
            'dev': dev
        }
        return next_obs, rewards, terminals, info

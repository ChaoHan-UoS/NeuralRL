import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

from sac.sac import SAC
from utils.mpc import LatentODE
from utils.running_stats_utils import RunningStats
from utils.utils import standardise
from utils.utils_data import get_inf_iterloader
from utils.utils_models import ODEBlock, get_grad_penalty
from utils.utils_train_sde import (
    plot_real_gen_signals, plot_cond_dist, plot_feature_latent, plot_feat_latent_time_cond
)

# %%
class SDEFuncLatent(nn.Module):
    def __init__(
            self,
            f_model: ODEBlock | LatentODE,
            g_out_dim: int,                         # state dim/latent dim
            g_model: nn.Sequential,
            g_proj_model: nn.Linear | None = None,  # SDE state/latent -> general state/latent
            use_latent: bool = False
    ):
        super().__init__()
        self.f_model = f_model
        self.g_model = g_model
        self.g_proj_model = g_proj_model
        self.g_out_dim = g_out_dim
        self.use_latent = use_latent

    def proj_g(self, x):
        return self.g_proj_model(x) if self.g_proj_model is not None else x

    def g(self, x0, in_signal, latent=None, t=None):
        """
        Args:
            x0 (torch.Tensor): (B, F_s)
            in_signal (torch.Tensor): (B, F_a)
            latent (torch.Tensor) | None: (B, Latent_Dim)
            t (torch.Tensor) | None: scalar
        """
        assert (latent is not None) == self.use_latent
        parts = [in_signal, x0]
        if self.use_latent:
            parts.append(latent)
        s = torch.concatenate(parts, dim=-1)
        return self.g_model(s)


class SDEBlockLatent(nn.Module):
    def __init__(
            self,
            func: SDEFuncLatent,
            latent_2_state_proj,
            dt,
            min_values,
            max_values,
            use_clip=True,
            noise_type='diagonal',
    ):
        super().__init__()
        self.func = func
        self.drift_func = self.func.f_model  # ODEBlock or LatentODE
        self.diff_func = self.func.g
        self.register_buffer('dt', dt)
        self.noise_type = noise_type
        self.use_latent = self.func.use_latent
        self.g_out_dim = self.func.g_out_dim

        if self.use_latent:
            self.latent_2_state_proj=latent_2_state_proj
        self.dw_dist = torch.distributions.Normal(
            loc=0, scale=torch.sqrt(self.dt)
        )

        # Used to clip values
        self.register_buffer('min_values', min_values)
        self.register_buffer('max_values', max_values)
        self.use_clip = use_clip

    def forward(self, ts, in_signal, x0):
        """
        Args:
            ts (torch.Tensor): (T, )
            actions (torch.Tensor): (B, F_a, T)
            x0 (torch.Tensor): (B, F_s)
        Returns: (torch.Tensor): (B, F_s, T)
        """
        z_list = [None]
        if self.use_latent:
            z0 = torch.randn(x0.shape[0], self.drift_func.latent_dim).to(x0)
            z_list = [z0]

        x_list = [x0]
        noise_shape = self.get_noise_shape(x0)

        T = in_signal.shape[-1]
        t = ts[0]
        for n in range(1, T):
            a_t = in_signal[..., n - 1]
            x_t = x_list[n - 1]
            z_t = z_list[n - 1]

            x_tp1, z_tp1 = self.get_next_state(a_t, x_t, z_t, t, noise_shape)
            x_list.append(x_tp1)
            z_list.append(z_tp1)
            t = t + self.dt
        return torch.stack(x_list, dim=-1)

    def get_noise_shape(self, x_t):
        noise_shape = (x_t.shape[0], self.g_out_dim) \
            if self.noise_type == 'diagonal' else (x_t.shape[0], 1)
        return noise_shape

    def get_next_state(self, a_t, x_t, z_t=None, t=None, noise_shape=None):
        if noise_shape is None:
            noise_shape = self.get_noise_shape(x_t)

        if self.use_latent:
            drift_z = self.drift_func.get_next_latent(a_t, z_t, x_t)
            diff_z = self.diff_func(x_t, a_t, z_t, t=t) * self.dw_dist.sample(noise_shape).to(self.dt.device)
            z_tp1 = self.func.proj_g(drift_z + diff_z)
            x_tp1 = self.latent_2_state_proj(z_tp1)
        else:
            drift_x = self.drift_func.get_next_state(a_t, x_t)
            diff_x = self.diff_func(x_t, a_t, t=t) * self.dw_dist.sample(noise_shape).to(self.dt.device)
            x_tp1 = self.func.proj_g(drift_x + diff_x)
            z_tp1 = None

        if self.use_clip:
            x_tp1 = torch.clamp(x_tp1, self.min_values, self.max_values)
        return x_tp1, z_tp1


class CollapseDiscWLayerNorm(nn.Module):
    def __init__(self, f_s=4, f_a=1, traj_len=20, mlp_size=100, use_action=False):
        super().__init__()
        self.feat_model = nn.Sequential(
            nn.Linear(f_s * traj_len if not use_action else (f_s + f_a) * traj_len, mlp_size),
            nn.LayerNorm(mlp_size),
            nn.Tanh(),
            nn.Linear(mlp_size, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.use_action = use_action

    def forward(self, x: torch.Tensor, in_signal=None):
        # x: (B, F_s, T)
        # in_signal: (B, F_a, T)
        if not self.use_action:
            s = self.feat_model(x.reshape(x.shape[0], -1))
        else:
            s = torch.concatenate([in_signal, x], dim=1)
            s = self.feat_model(s.reshape(x.shape[0], -1)).squeeze(-1)
        return s

# %%
class ReplayMemory(object):
    # Storing tensor data
    def __init__(self, capacity, tuple):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0
        self.tuple = tuple

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.tuple(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # Can convert cpu tensor to array
        return map(np.stack, zip(*batch))
    
    def sample_tensor(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return map(torch.stack, zip(*batch))

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


Trajectory = namedtuple(
    'Trajectory',
    ('states', 'actions')
)
TrajectoryPolicy = namedtuple(
    'TrajectoryPolicy',
    ('states', 'actions', 'rnd')  # rnd is True if using random policy to collect the traj
)
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'mask')
)
TransitionLatent = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'mask', 'latent', 'next_latent')
)


def add_tensor_to_replay(dataset, memory, f_s):
    for i in range(dataset.shape[0]):
        obj = dataset[i]
        state = obj[..., :f_s]
        action = obj[..., f_s:]
        memory.push(state, action, torch.tensor(False))

# %%
class MBRL:
    def __init__(
            self,
            env_model: ODEBlock | LatentODE | None,
            agent: SAC,
            memory_traj_train: ReplayMemory,
            memory_traj_val: ReplayMemory,
            memory_trans: ReplayMemory,
            env,
            f_s: int,
            f_a: int,
            std_act = False,     # Standardize action
            latent_m = False,    # Use latent model
            latent_p = False,    # Use latent policy
            TF = True,           # Teacher forcing for training latent ODE decoder
            KL_COEFF = .99,
            latent_dim = 10,
            device = 'cuda'
    ):
        self.env_model = env_model
        self.agent = agent
        self.memory_traj_train = memory_traj_train
        self.memory_traj_val = memory_traj_val
        self.memory_trans = memory_trans
        self.curr_env_steps = 0
        self.update_steps = 0    # Agent update steps over E epochs
        self.env = env
        self.model_g: SDEBlockLatent = None
        self.model_d = None

        self.F_S = f_s
        self.F_A = f_a
        self.std_act = std_act
        self.device = device

        # Data normalisation
        self.rms = RunningStats(device="cpu", dim=self.F_S)
        self.states_mean = None
        self.states_std = None
        self.actions_mean = None
        self.actions_std = None

        self.latent_m = latent_m
        self.latent_p = latent_p
        self.TF = TF
        self.KL_COEFF = KL_COEFF
        # Standard Gaussian prior for z0; its sample z0 is used
        # to generate latent traj at val (or planning) time
        self.prior_z0 = torch.distributions.Normal(
            torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device)
        )
        self.latent_dim = latent_dim

    # ---------------------- ODE train -----------------------------
    def train_step(self, optimizer, ts, actions, states):
        """
        Single optim step of ODE model on one batch of trajs
        - ts:      (T, )
        - actions: (B, F_a, T)
        - states:  (B, F_s, T)
        """
        if self.latent_m:
            # For latent ODE:
            optimizer.zero_grad()
            mu, log_var = self.env_model.encode(
                actions[..., :-1], states[..., :-1]
            )  # (B, Latent_Dim)
            z0 = self.env_model.sample_latent(mu, log_var)  # (B, Latent_Dim)
            pred_next_states = self.env_model.get_next_states(
                actions[..., :-1],
                z0,
                states if self.TF else states[..., 0],
                tf=self.TF
            )  # (B, F_s, T - 1), s_1, ..., s_{T-1}

            # Analytic KL and rec error in negative ELBO loss,
            # assuming diagonal-Gaussian posterior and standard Gaussian prior
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            rec_loss = torch.nn.functional.mse_loss(pred_next_states, states[..., 1:])
            loss = rec_loss + self.KL_COEFF * kl_loss.mean()
        else:
            # For neural ODE:
            optimizer.zero_grad()
            out = self.env_model(ts, actions, states[..., 0])
            loss = torch.nn.functional.mse_loss(out, states)
        loss.backward()
        optimizer.step()

        return rec_loss if self.latent_m else loss

    def train_loop_from_dataset(self, train_loader, optimizer, ts, epoch):
        # One epoch of ODE model training over all batches
        loss_acc = 0
        for states, actions, _ in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)

            loss = self.train_step(optimizer, ts, actions, states)
            loss_acc += loss.item()
        loss_acc /= len(train_loader)

        if wandb.run is not None:
            wandb.log(dict(ODE_train_loss=loss_acc, ODE_epoch=epoch))

        return loss_acc

    @torch.no_grad()
    def val_step(self, ts, actions, states):
        """
        Single validation step of ODE model on one batch of trajs (no gradient)
        - ts:      (T, )
        - actions: (B, F_a, T)
        - states:  (B, F_s, T)
        """
        if self.latent_m:
            z0 = self.prior_z0.sample(
                torch.Size((states.shape[0], self.latent_dim))
            ).squeeze(-1).to(states.device)  # (B, Latent_Dim)
            pred_next_states = self.env_model.get_next_states(
                actions[..., :-1],
                z0,
                states[..., 0]
            )  # (B, F_s, T - 1), s_1, ..., s_{T-1}
            val_rec_loss = torch.nn.functional.mse_loss(pred_next_states, states[..., 1:])
        else:
            out = self.env_model(ts, actions, states[..., 0])
            loss = torch.nn.functional.mse_loss(out, states)

        return val_rec_loss if self.latent_m else loss

    @torch.no_grad()
    def val_loop_from_dataset(self, val_loader, ts, epoch):
        # One epoch of ODE model validation over all batches
        loss_acc = 0
        for states, actions, _ in val_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)

            loss = self.val_step(ts, actions, states)
            loss_acc += loss.item()
        loss_acc /= len(val_loader)

        if wandb.run is not None:
            wandb.log(dict(ODE_val_loss=loss_acc, ODE_epoch=epoch))

        return loss_acc

    def train_ode_loop(
            self, optimizer, ts, total_ode_epochs=0, n_epochs=500, batch_size=256, rms=True,
            early_stopper=None
    ):
        # ((N, F_s, T), (N, F_a, T), (N, ))
        train_dataset = TensorDataset(
            *self.read_trajs_from_buffer(self.memory_traj_train, rms=rms)
        )
        val_dataset = TensorDataset(
            *self.read_trajs_from_buffer(self.memory_traj_val, rms=rms)
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        print('Training ODE')
        self.env_model.requires_grad_(True)
        curr_ode_epochs = 0
        max_ode_epochs = total_ode_epochs + n_epochs
        for env_model_epoch in range(total_ode_epochs, max_ode_epochs):
            # MSE rec loss at each epoch
            loss = self.train_loop_from_dataset(train_loader, optimizer, ts, env_model_epoch)
            val_loss = self.val_loop_from_dataset(val_loader, ts, env_model_epoch)
            print(f'Epoch {env_model_epoch} Train Loss: {loss:.6f} Val Loss: {val_loss:.6f}')
            curr_ode_epochs += 1

            if early_stopper is not None and early_stopper.early_stop(val_loss, self.env_model):
                print('Early stopping')
                # Restore the best ODE model
                self.env_model.load_state_dict(early_stopper.best_model_state)
                min_val_loss = early_stopper.min_validation_loss
                print(f'Best val loss: {min_val_loss:.6f}')
                break

        print(f"No improvement for {early_stopper.counter} epochs")
        print(f'ODE trained for: {curr_ode_epochs} epochs')
        return curr_ode_epochs

    def read_trajs_from_buffer(self, memory, rms=False, transpose=True):
        # Read all trajs from current replay memory
        # (N, T, F_s), (N, T, F_a), (N, )
        states_N, actions_N, source_N = memory.sample_tensor(len(memory))
        states_N = self.stand_states(states_N, rms)
        actions_N = self.stand_actions(actions_N)
        if transpose:
            states_N = states_N.transpose(-1, -2)
            actions_N = actions_N.transpose(-1, -2)
        return states_N, actions_N, source_N

    # ---------------------- SDE train -----------------------------
    def train_sde(
            self, REG_TERM, N_CRITIC, ts, optimizer_d, optimizer_g, epoch, batch_size=256, rms=True
    ):
        assert self.model_g is not None
        assert self.model_d is not None
        
        self.model_g.train()
        self.model_d.train()

        train_dataset = TensorDataset(
            *self.read_trajs_from_buffer(
                self.memory_traj_train, rms=False, transpose=False
            )
        )
        train_loader = get_inf_iterloader(
            DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        )

        for _ in range(N_CRITIC):
            ########################
            # Train Discriminator
            ########################
            target, actions, source = next(train_loader)
            with torch.no_grad():
                actions = self.stand_actions(actions)
                target = self.stand_states(target)

            target = target.transpose(-1, -2)
            y0 = target[..., 0].to(self.device)
            target = target.to(self.device)
            actions = actions.transpose(-1, -2).to(self.device)

            optimizer_d.zero_grad()
            real_score = self.model_d(target, actions)
            D_real_loss = -torch.mean(real_score)

            generated_samp = self.model_g(ts, actions, y0)
            generated_score = self.model_d(generated_samp, actions)
            D_fake_loss = torch.mean(generated_score)

            # Calculating the negative loss of the discriminator: gen_score - real_score
            grad_penalty = get_grad_penalty(
                target, generated_samp, self.model_d, in_signal=actions, reg_param=REG_TERM
            )
            D_loss = torch.mean(D_fake_loss + D_real_loss) + grad_penalty
            # D_loss = torch.mean(D_fake_loss + D_real_loss)

            D_loss.backward()
            optimizer_d.step()

        ########################
        # Train Generator
        ########################
        optimizer_g.zero_grad()

        gen_samp = self.model_g(ts, actions, y0)
        G_loss = -torch.mean(self.model_d(gen_samp, actions))
        G_loss.backward()
        optimizer_g.step()

        if wandb.run is not None:
            wandb.log(
                {
                    "train_D_loss": D_loss.item(),
                    "train_G_loss": G_loss.item(),
                    "train_grad_pen": grad_penalty.item(),
                    "W_dist_train": (D_loss - grad_penalty).item(),
                    "SDE_epoch": epoch,
                }
            )

    def train_sde_loop(
            self, optimizer_d, optimizer_g, ts, ts_, total_sde_epochs=0, n_epochs=5000,
            batch_size=128, rms=True, verbose=False
    ):
        curr_sde_epochs = 0

        min_values, max_values = self.get_min_max()
        self.model_g.register_buffer('min_values', min_values)
        self.model_g.register_buffer('max_values', max_values)

        # Load trained ode and freeze it
        self.model_g.func.f_model.load_state_dict(self.env_model.state_dict())
        self.model_g.func.f_model.requires_grad_(False)
        self.model_g.to(self.device)

        print('Training SDE')
        max_sde_epochs = total_sde_epochs + n_epochs
        for env_model_epoch in range(total_sde_epochs, max_sde_epochs):
            self.train_sde(
                REG_TERM=10,
                N_CRITIC=1,
                ts=ts,
                optimizer_d=optimizer_d,
                optimizer_g=optimizer_g,
                epoch=env_model_epoch,
                batch_size=batch_size,
                rms=rms
            )
            curr_sde_epochs += 1
        if n_epochs != 0:
            self.visualize(ts, ts_, epoch=env_model_epoch, verbose=verbose)

        print(f'SDE trained for: {curr_sde_epochs} epochs')
        return curr_sde_epochs

    # -------------------- visualize model train -----------------------
    @torch.no_grad()
    def set_visualization_datataset(self, vis_size=200, rms=True):
        vis_size = min(self.memory_traj_val.position, vis_size)
        states, actions, _ = self.memory_traj_val.sample_tensor(vis_size)
        states = self.stand_states(states, rms)
        actions = self.stand_actions(actions)
        states = states.transpose(-1, -2)
        actions = actions.transpose(-1, -2)

        self.vis_x0s = states[..., 0].to(self.device)
        self.vis_actions = actions.to(self.device)
        self.vis_feature_ = states.numpy(force=True)
        self.vis_times = [1, 5, 9]

    @torch.no_grad()
    def visualize(self, ts, ts_, epoch, verbose=True):
        # vis_samp_: B, F_S, T
        vis_samp_ = self.model_g(ts, self.vis_actions, self.vis_x0s).numpy(force=True)
        plot_real_gen_signals(
            self.vis_feature_, vis_samp_, ts_, epoch=epoch, n_samples=1000,
            n_states=self.F_S, verbose=verbose,
        )
        plot_cond_dist(
            self.vis_times, self.vis_feature_, vis_samp_, epoch=epoch,
            n_states=self.F_S, verbose=verbose,
        )

        vis_feat_hat_ = self.model_g(ts, self.vis_actions, self.vis_x0s).numpy(force=True)
        plot_feature_latent(
            self.vis_feature_, vis_feat_hat_, epoch=epoch, tsne=False,
            n_states=self.F_S, verbose=verbose,
        )
        plot_feature_latent(
            self.vis_feature_, vis_feat_hat_, epoch=epoch, tsne=True,
            n_states=self.F_S, verbose=verbose,
        )
        plot_feat_latent_time_cond(
            self.vis_times, self.vis_feature_[:1000], vis_feat_hat_[:1000], epoch=epoch,
            tsne=False, verbose=verbose,
        )
        plot_feat_latent_time_cond(
            self.vis_times, self.vis_feature_[:1000], vis_feat_hat_[:1000], epoch=epoch,
            tsne=True, verbose=verbose,
        )

    def plot_init_pred(self, ts_):
        pred_ = self.env_model(self.vis_actions, self.vis_x0s).numpy(force=True)
        plot_real_gen_signals(
            self.vis_feature_, pred_, ts_, n_samples=1000, epoch=None, verbose=False, n_states=self.F_S
        )
        plot_cond_dist(
            self.vis_times, self.vis_feature_, pred_, epoch=-1, verbose=False, n_states=self.F_S
        )

    # ---------------------------- mpc  -----------------------------
    @torch.no_grad()
    def mpc_search(self, init_state, h, k, e=50, combine_mf=True, rms=True):
        """
        Perform MPC search to find the best action sequence.

        Args:
            init_state (List[torch.Tensor, torch.Tensor]): [normalized state, latent state]
            h (int): The planning horizon.
            k (int): The number of action sequences to sample.
            e (int, optional): The number of top action sequences to consider for averaging.
            Default is 50.
            combine_mf (bool, optional): Whether to combine model-based and model-free approaches.
            Default is True.
            rms (bool, optional): Whether to use running mean and standard deviation for normalization.
            Default is True.
        Returns:
            np.ndarray: The best action sequence found by the MPC search.
        """
        actions_list = torch.empty(
            k, h, self.F_A, dtype=torch.float, device=self.device
        )  # (K, H, D_action)
        states_list = torch.empty(
            k, h+1, self.F_S, dtype=torch.float, device=self.device
        )  # (K, H+1, D_state)
        states_list[:, 0, :] = init_state[0].repeat(k, 1)

        if self.latent_m:
            latent_states_list = torch.empty(
                k, h+1, self.latent_dim, dtype=torch.float, device=self.device
            )
            latent_states_list[:, 0, :] = init_state[1].repeat(k, 1)

        for i in range(h):
            if combine_mf:
                obs_n_latent = torch.concatenate(
                    [states_list[:, i], latent_states_list[:, i]], -1
                ) if self.latent_p else states_list[:, i]
                actions_list[:, i, :] = self.agent.policy.sample(obs_n_latent)[0]
            else:
                actions_list[:, i, :].uniform_(-1, 1)

            at = actions_list[:, i, :]
            st = states_list[:, i, :].to(self.device)

            if self.latent_m:
                latent_state = latent_states_list[:, i, :].to(self.device)
                if self.model_g is None:
                    # Latent ODE
                    next_latent_states = self.env_model.get_next_latent(at, latent_state, st)
                    states_list[:, i + 1, :] = self.env_model.latent_2_state_proj(next_latent_states)
                    latent_states_list[:, i + 1, :] = next_latent_states
                else:
                    # Latent SDE
                    states_list[:, i + 1, :], latent_states_list[:, i + 1, :] = (
                        self.model_g.get_next_state(at, st, latent_state)
                    )
            else:
                norm_at = self.stand_actions(at.cpu()).to(self.device)
                if self.model_g is None:
                    # Neural ODE
                    states_list[:, i + 1, :] = self.env_model.get_next_state(norm_at, st)  # (K, D_state)
                else:
                    # Neural SDE
                    states_list[:, i + 1, :] = self.model_g.get_next_state(norm_at, st)[0]

        rewards_list = self.env.get_reward_batch(states_list, actions_list)  # (K, H)
        rewards = self.calc_acc_rewards(rewards_list)  # (K, )

        if combine_mf:
            # Concat state with latent state
            obs_n_latent = torch.concatenate(
                [states_list[:, -1, :], latent_states_list[:, -1, :]], -1
            ) if self.latent_p else states_list[:, -1, :]
            # For the final reward, use the sampled action if latent_m else determ action
            last_action = self.agent.policy.sample(obs_n_latent)[0 if self.latent_m else -1]
            last_states_value = self.calc_value(obs_n_latent, last_action)
            rewards += (self.agent.gamma ** h * last_states_value[..., 0])

        best_actions = actions_list[torch.topk(rewards, k=e, sorted=False)[1], 0, :]  # soft greedy
        return best_actions.mean(dim=0).cpu().numpy(force=True)

    def calc_value(self, state, action):
        q1, q2 = self.agent.critic(state, action)
        return torch.minimum(q1, q2)

    def calc_acc_rewards(self, rewards):
        # rewards: (B, T)
        time_steps = torch.arange(rewards.shape[1], device=rewards.device).repeat(rewards.shape[0], 1)
        discounts = self.agent.gamma ** time_steps
        if len(rewards.size()) == 1:  # (T, )
            return torch.dot(rewards, discounts)  # scalar
        elif len(rewards.size()) == 2:  # (B, T)
            return torch.mm(rewards, discounts.t()).diag()  # (B, )
        else:
            raise ValueError("rewards should be 1D vector or 2D matrix.")

    def mpc_planning(
            self,
            updates,           # Update count for target critic: 0 = update every time critic updates
            max_env_steps,     # Max episodic steps, adaptive when less than max_steps
            cut_len: int,
            batch_size=128,
            k=50,
            h=10,
            e=50,
            combine_mf=True,
            rand=False,        # Random action or not
            val_ratio=0,       # A prob ratio for saving traj to val or train buffer
            rms=True,          # For state only
            save_trans=True,
            save_trajs=True,
            mode='mb'
    ):
        # Reset the env and init variables
        states_lst = [torch.from_numpy(self.env.reset()[0])]  # [(F_S, )]
        actions_lst = []
        if self.latent_m:
            latent_states_lst = [self.env_model.get_init_latent().to(self.device)]  # [(1, Latent_Dim)]
        rewards = []
        length = 0

        for i in range(max_env_steps):
            # Normalize state
            state = states_lst[-1][None]  # (1, F_S)
            if rms:
                self.rms += state.squeeze(0)
            norm_state = self.stand_states(state, rms=rms).to(self.device)
            latent_state = latent_states_lst[-1].to(self.device) if self.latent_m else None

            # Select action
            if rand:
                action = np.random.uniform(-1, 1, size=self.F_A).astype(np.float32)
            elif mode == 'mb':
                # Find the best action at each env step
                action = self.mpc_search(
                    init_state=[norm_state, latent_state],
                    h=h,
                    k=k,
                    e=e,
                    combine_mf=combine_mf,
                    rms=rms,
                )
            elif mode == 'mf':
                action = self.agent.select_action(norm_state if rms else state)
            else:
                raise NotImplementedError(f'model must be either mf or mb but is now: {mode}')

            # Execute 1 env step and save transitions for agent training
            with torch.no_grad():
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state)
                action = torch.from_numpy(action[None]).to(self.device)  # (1, F_A)
                next_latent_state = self.env_model.get_next_latent(
                    action, latent_state, norm_state
                ) if self.latent_m else None

                if save_trans:
                    self.save_trans_(
                        state, action, next_state, reward, done,
                        latent_state, next_latent_state
                    )
                states_lst.append(next_state)
                actions_lst.append(action.squeeze(dim=0).cpu())
                rewards.append(reward)
                if self.latent_m:
                    latent_states_lst.append(next_latent_state.cpu())
                self.curr_env_steps += 1
                length += 1

            # Update agent at each env step if the episode does not terminate
            if done or truncated:
                break
            elif len(self.memory_trans) > batch_size and combine_mf:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = (
                    self.agent.update_parameters(
                        self.memory_trans,
                        batch_size,
                        updates,
                        standardiser=(self.rms.m, self.rms.std) if rms else None,
                        latent=self.latent_p
                    )
                )
                self.update_steps += 1

        # Save trajs for model training
        if save_trajs:
            self.save_trajs_(cut_len, rand, val_ratio, states_lst, actions_lst)

        # Return accumulated rewards and length of a traj
        return sum(rewards), length

    def save_trans_(
            self, state, action, next_state, reward, done, latent_state=None,
            next_latent_state=None
    ):
        state = state.cpu().squeeze(0)
        action = action.cpu().squeeze(0)
        next_state = next_state.cpu()
        mask = float(not done)

        if self.latent_p:
            latent_state = latent_state.cpu().squeeze(0)
            next_latent_state = next_latent_state.cpu().squeeze(0)
            self.memory_trans.push(
                state, action, reward, next_state, mask, latent_state, next_latent_state
            )
        else:
            self.memory_trans.push(
                state, action, reward, next_state, mask
            )

    def save_trajs_(self, cut_len, rand, val_ratio, states_lst, actions_lst):
        states = torch.stack(states_lst)    # (L + 1, F_S)
        actions = torch.stack(actions_lst)  # (L, F_A)

        # Split into non-overlapping chunks of length <=cut_len
        states_chunks = states.split(cut_len, 0)    # ((<=cut_len, F_S))
        actions_chunks = actions.split(cut_len, 0)

        # Keep only the chunks whose length == cut_len
        states_filter = filter(lambda x: x.shape[0] == cut_len, states_chunks)
        actions_filter = filter(lambda x: x.shape[0] == cut_len, actions_chunks)

        # Handle the chunk remainder by grabbing the last cut_len window
        self._get_last_states_actions(cut_len, rand, states, actions)

        self.save_traj_to_replay(
            states_filter,
            actions_filter,
            rnd=rand,
            val=True if np.random.rand() < val_ratio else False
        )

    def _get_last_states_actions(self, cut_len, rand, states, actions):
        # Only if L is not divisible by cut_len and L >= cut_len
        if actions.shape[0] % cut_len != 0 and actions.shape[0] >= cut_len:
            # Create a single‐element iterator containing that last cut_len window
            remainder_states_iter = iter([states[-cut_len-1: -1]])
            remainder_actions_iter = iter([actions[-cut_len:]])
            self.save_traj_to_replay(remainder_states_iter, remainder_actions_iter, rnd=rand)

    def save_traj_to_replay(self, states_filter, actions_filter, rnd, val=False):
        # save trajs for model train
        for (states, actions) in zip(states_filter, actions_filter):
            # (T, F_S), s_0, ..., s_{T-1}
            # (T, F_A), a_0, ..., a_{T-1}
            assert states.shape[1] == self.F_S, 'Wrong state shape whilst saving trajs'
            assert actions.shape[1] == self.F_A, 'Wrong action shape whilst saving trajs'
            if val:
                self.memory_traj_val.push(
                    states.cpu(), actions.cpu(), torch.tensor(rnd)
                )
            else:
                self.memory_traj_train.push(
                    states.cpu(), actions.cpu(), torch.tensor(rnd)
                )

    # ---------------------------- data process  -----------------------------
    @torch.no_grad()
    def stand_states(self, states, rms=True):
        return self.rms.normalize(states) if rms else states

    @torch.no_grad()
    def stand_actions(self, act):
        return standardise(act, self.actions_mean, self.actions_std) if self.std_act else act

    def get_min_max(self):
        # Compute the min and max of each feature over N and T
        all_data = torch.stack(
            list(
                map(lambda x : x.states, self.memory_traj_train.memory)
            )
        )  # (N, T, F_S)
        all_data = self.stand_states(all_data)
        min_per_feature = all_data.min(0)[0].min(0)[0]  # (F_S, )
        max_per_feature = all_data.max(0)[0].max(0)[0]  # (F_S, )
        return min_per_feature, max_per_feature

    def calc_mean_std(self):
        # (N, T, F_S), (N, T, F_A)
        states, actions, _ = map(torch.stack, zip(*self.memory_traj_train.memory))
        self.states_mean = states.mean((0, 1))
        self.states_std = states.std((0, 1))
        self.actions_mean = actions.mean((0, 1))
        self.actions_std = actions.std((0, 1))

    # ---------------------------- others  -----------------------------
    @torch.no_grad()
    def eval_agent(self, n_episodes, rms=True):
        # (Latent_Dim, )
        latent_state = self.prior_z0.sample(torch.Size((self.latent_dim, ))).squeeze(-1) \
            if self.latent_p else None
        reward_lst = []
        for _ in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            truncated = False
            episode_reward = 0

            while not done and not truncated:
                state = self.stand_states(torch.from_numpy(state), rms=rms).to(self.device)
                latent_state = latent_state.to(self.device) if self.latent_p else None
                action = self.agent.select_action(
                    torch.concat([state, latent_state], -1) if self.latent_p else state,
                    evaluate=True
                )
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                if self.latent_p:
                    action = torch.from_numpy(action)[None].to(self.device)
                    state = state[None]
                    latent_state = self.env_model.get_next_latent(
                        action, latent_state[None], state
                    ).squeeze(0)
                state = next_state

            reward_lst.append(episode_reward)

        reward_lst = np.array(reward_lst)
        return reward_lst

    def save_model(self, env_name, run_id):
        dir = Path("../checkpoints")
        dir.mkdir(exist_ok=True)
        model_name = f'{env_name}_lsde_trans_{run_id}.pt' if self.latent_m \
            else f'{env_name}_sde_trans_{run_id}.pt'
        model_path = dir / model_name

        torch.save({'state_dict_gen': self.model_g.state_dict(),
                    'state_dict_disc': self.model_d.state_dict()}, model_path)
        print(f"SDE Model Saved! {model_path}")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  # Number of consecutive epochs without improvement
        self.min_validation_loss = float('inf')
        self.best_model_state = None

    def less_than(self, validation_loss):
        return validation_loss < self.min_validation_loss

    def early_stop(self, validation_loss, env_model):
        """ Called at each epoch to determine whether training ODE should stop"""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = env_model.state_dict()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # Stop training if the validation loss does not improve
            # for `patience` consecutive epochs
            if self.counter >= self.patience:
                return True
        return False

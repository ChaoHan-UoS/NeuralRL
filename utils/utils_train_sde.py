import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb

from utils.utils import standardise, MLP, save_model_gan
from utils.utils_data import (
    TrajDataset, generate_dataset_parallel, get_inf_iterloader, get_mean_std
)
from utils.utils_models import (
    ODEBlock, ODEFunc, SDEBlock, SDEFunc, get_grad_penalty
)


wandb.login()


def get_state_names(n_states):
    if n_states == 4:
        return ["Position", "Velocity", "Pole Angle", "Pole Angular Velocity"]
    elif n_states == 3:
        return ["Velocity", "Pole Angle", "Pole Angular Velocity"]
    elif n_states == 1:
        return ["Velocity"]
    elif n_states == 2:
        return ["Pole Angle", "Pole Angular Velocity"]


def plot_real_gen_signals(
        target_, pred_, ts_, epoch=0, n_samples=100, n_states=4, verbose=False, comment=None
):
    # Plots real and generated signals for comparison
    # target_/pred_: numpy (B, F_s, T)
    target_ = target_[:n_samples]
    pred_ = pred_[:n_samples]
    target_ = np.swapaxes(target_, 0, 2)  # (T, F_s, B)
    pred_ = np.swapaxes(pred_, 0, 2)  # (T, F_s, B)
    # n_states = target_.shape[1]

    state_names = get_state_names_generic(n_states)
    for state_idx, title_str in zip(range(n_states), state_names):
        if epoch is None:
            title_str = f"{title_str} Initial"
        if comment is not None:
            title_str = f"{title_str} ({comment})"

        fig, ax = plt.subplots()
        ax.plot(ts_, target_[:, state_idx], color="tab:blue", alpha=0.3, label="Real")
        ax.plot(ts_, pred_[:, state_idx], color="tab:red", alpha=0.3, label="Generated")
        if epoch is None:
            ax.set_title(title_str)
        else:
            ax.set_title(f"{title_str} Epoch:{epoch + 1:0>3d}")

        ax.set_xlabel("Time")
        # Both handles and identical labels for "Real" (and similarly for "Generated")
        handle, label = ax.get_legend_handles_labels()
        # Avoid repeating identical labels in the legend
        handle = [handle[0], handle[-1]]
        label = [label[0], label[-1]]
        plt.legend(handle, label, loc="best")

        if wandb.run is not None:
            wandb.log({f"{title_str}": wandb.Image(fig), "epoch": epoch})

    if verbose:
        plt.show()
    else:
        plt.close("all")


def get_state_names_generic(n_states):
    return [f'Feature {i}' for i in range(n_states)]


def plot_cond_dist(
        vis_times, target_, pred_, pred_f_=None, hist_n_bins=25, epoch=0, n_states=4,
        verbose=False
):
    # Plots the distribution of real and generated signals at specified time points
    # target_/pred_/pred_f_: numpy (N, F_s, T)
    target_ = np.swapaxes(target_, 0, 2)  # (T, F_s, N)
    pred_ = np.swapaxes(pred_, 0, 2)  # (T, F_s, N)
    if pred_f_ is not None:
        pred_f_ = np.swapaxes(pred_f_, 0, 2)  # (T, F_s, N)

    state_names = get_state_names_generic(n_states)
    for state_idx, title_str in zip(range(n_states), state_names):
        fig, axs = plt.subplots(1, len(vis_times), figsize=(5 * len(vis_times), 5))
        for ax, t in zip(axs, vis_times):
            ax.hist(
                target_[t, state_idx], color="blue", alpha=0.5, label="Real",
                density=True, bins=hist_n_bins
            )
            ax.hist(
                pred_[t, state_idx], color="red", alpha=0.5, label="SDE-Generated",
                density=True, bins=hist_n_bins
            )
            if pred_f_ is not None:
                ax.hist(
                    pred_f_[t, state_idx], color="green", alpha=0.5, label="ODE-Generated",
                    density=True, bins=hist_n_bins
                )
            ax.set_title(f"T={t + 1}")
        plt.legend(loc="best")
        fig.suptitle(f"{title_str} Epoch:{epoch:0>3d}")
        fig.tight_layout()

        if pred_f_ is not None:
            title_str += " + Initial"

        if wandb.run is not None:
            wandb.log({f"{title_str} Dist": wandb.Image(fig), "epoch": epoch})

    if verbose:
        plt.show()
    else:
        plt.close("all")


def plot_global_latent_space(B, vis_data_, vis_hat_, epoch=0, tsne=True, verbose=False):
    if tsne:
        tsne_solver = TSNE(perplexity=40, max_iter=300)
        tsne_input = np.concatenate([vis_data_, vis_hat_], axis=0)
        tse_out = tsne_solver.fit_transform(tsne_input)
        target = tse_out[:B]
        generated = tse_out[B:]

        fig, ax = plt.subplots()
        ax.scatter(
            generated[:, 0], generated[:, 1], label="Generated", color="blue", alpha=0.4
        )
        ax.scatter(target[:, 0], target[:, 1], label="Real", color="red", alpha=0.3)
        ax.set_title(f"t-SNE Epoch: {epoch:0>3d}")
        plt.legend()
        if wandb.run is not None:
            wandb.log({"tSNE": wandb.Image(fig), "epoch": epoch})
    else:
        pca_solver = PCA(n_components=2)
        target = pca_solver.fit_transform(vis_data_)
        generated = pca_solver.transform(vis_hat_)

        fig, ax = plt.subplots()
        ax.scatter(
            generated[:, 0], generated[:, 1], label="Generated", color="blue", alpha=0.3
        )
        ax.scatter(target[:, 0], target[:, 1], label="Real", color="red", alpha=0.3)
        ax.set_title(f"PCA Epoch: {epoch:0>3d}")
        plt.legend()
        if wandb.run is not None:
            wandb.log({"PCA": wandb.Image(fig), "epoch": epoch})

    if verbose:
        plt.show()
    else:
        plt.close("all")


def plot_feature_latent(
        vis_feature_, vis_feat_hat_, epoch=0, tsne=True, n_states=4, verbose=False
):
    # Plots the latent space (B, T -> B, 2) using t-SNE or PCA
    # vis_feature_/vis_feat_hat_: numpy (B, F_s, T)
    B = vis_feature_.shape[0]
    state_names = get_state_names_generic(n_states)
    for feat_idx, feat_name in zip(range(n_states), state_names):
        if tsne:
            tsne_solver = TSNE(perplexity=40, max_iter=300)
            tsne_input = np.concatenate(
                [vis_feature_[:, feat_idx], vis_feat_hat_[:, feat_idx]], axis=0
            )  # (2B, T)
            tse_out = tsne_solver.fit_transform(tsne_input)  # (2B, 2), joint latent embedding
            target = tse_out[:B]
            generated = tse_out[B:]

            fig, ax = plt.subplots()
            ax.scatter(
                target[:, 0], target[:, 1], label="Real", color="blue", alpha=0.5
            )
            ax.scatter(
                generated[:, 0], generated[:, 1], label="Generated", color="red", alpha=0.5
            )
            ax.set_title(f"t-SNE {feat_name} Epoch: {epoch:0>3d}")
            plt.legend()
            if wandb.run is not None:
                wandb.log({f"tSNE {feat_name}": wandb.Image(fig), "epoch": epoch})
        else:
            pca_solver = PCA(n_components=2)
            target = pca_solver.fit_transform(vis_feature_[:, feat_idx])  # (B, 2)
            # Apply the same transformation by using .transform()
            generated = pca_solver.transform(vis_feat_hat_[:, feat_idx])  # (B, 2)

            fig, ax = plt.subplots()
            ax.scatter(
                target[:, 0], target[:, 1], label="Real", color="blue", alpha=0.5
            )
            ax.scatter(
                generated[:, 0], generated[:, 1], label="Generated", color="red", alpha=0.5
            )
            ax.set_title(f"PCA {feat_name} Epoch: {epoch:0>3d}")
            plt.legend()
            if wandb.run is not None:
                wandb.log({f"PCA {feat_name}": wandb.Image(fig), "epoch": epoch})

    if verbose:
        plt.show()
    else:
        plt.close("all")


def plot_feat_latent_time_cond(
        vis_times, vis_feature_, vis_feat_hat_, epoch=0, tsne=True, verbose=False
):
    # Plots the latent space (B, F_s -> B, 2) at specified time points using t-SNE or PCA
    fig, axs = plt.subplots(1, len(vis_times), figsize=(5 * len(vis_times), 5))
    B = vis_feat_hat_.shape[0]
    if tsne:
        tsne_solver = TSNE(perplexity=40, max_iter=300)
        for ax, time_idx in zip(axs, vis_times):
            tsne_input = np.concatenate(
                [vis_feature_[..., time_idx], vis_feat_hat_[..., time_idx]], axis=0
            )  # (2B, F_s)
            tse_out = tsne_solver.fit_transform(tsne_input)  # (2B, 2), joint latent embedding
            target = tse_out[:B]
            generated = tse_out[B:]

            ax.scatter(
                target[:, 0], target[:, 1], label="Real", color="blue", alpha=0.5
            )
            ax.scatter(
                generated[:, 0], generated[:, 1], label="Generated", color="red", alpha=0.5
            )
            ax.set_title(f"T={time_idx + 1}")
    else:
        pca_solver = PCA(n_components=2)
        for ax, time_idx in zip(axs, vis_times):
            target = pca_solver.fit_transform(vis_feature_[..., time_idx])  # (B, 2)
            # Apply the same transformation by using .transform()
            generated = pca_solver.transform(vis_feat_hat_[..., time_idx])  # (B, 2)

            ax.scatter(
                target[:, 0], target[:, 1], label="Real", color="blue", alpha=0.5
            )
            ax.scatter(
                generated[:, 0], generated[:, 1], label="Generated", color="red", alpha=0.5,
            )
            ax.set_title(f"T={time_idx + 1}")

    if tsne:
        title_str = "TSNE (Time)"
    else:
        title_str = "PCA (Time)"
    plt.legend()
    fig.suptitle(f"{title_str} Epoch:{epoch:0>3d}")
    fig.tight_layout()

    if wandb.run is not None:
        wandb.log({title_str: wandb.Image(fig), "epoch": epoch})

    if verbose:
        plt.show()
    else:
        plt.close("all")


class CollapseDisc(nn.Module):
    def __init__(self, traj_len=20, f_s=4, f_a=1, num_layers=2, mlp_size=100):
        super().__init__()
        self.feat_model = MLP(
            in_size=(f_s + f_a) * traj_len,
            mlp_size=mlp_size,
            out_size=1,
            num_layers=num_layers,
            tanh=True,
            selu=False,
        )

    def forward(self, x, in_signal):
        # x : (B, F_s, T)
        # in_signal : (B, F_a, T)
        s = torch.concatenate([in_signal, x], dim=1)
        s = self.feat_model(s.view(s.shape[0], -1)).squeeze(-1)
        # (B, )
        return s


class SDEBlockManualClipping(SDEBlock):
    def __init__(self, func, dt, min_lim_vel, max_lim_vel, method="euler"):
        super().__init__(func, dt, method)
        self.drift_func = func.f
        self.diff_func = func.g

        # Limit Tensor
        self.max_lim_tensor = nn.Parameter(torch.ones(4), requires_grad=False)
        self.max_lim_tensor *= torch.inf
        self.max_lim_tensor[1] = max_lim_vel

        self.min_lim_tensor = nn.Parameter(torch.ones(4), requires_grad=False)
        self.min_lim_tensor *= -torch.inf
        self.min_lim_tensor[1] = min_lim_vel

    def clip(self, x):
        return torch.clamp(x, self.min_lim_tensor, self.max_lim_tensor)

    def forward(self, ts, actions, y0, noise):
        """
        Args:
            ts (torch.Tensor): T
            actions (torch.Tensor): [B, F_a, T]
            y0 (torch.Tensor): [B, F_s]
            noise (torch.Tensor): [B, F_s] or [B, F_s, T]
        """
        # ts: T
        # Actions: [B, F_a, T]
        # y0: [B, F]
        # Create the for loop here
        T = ts.shape[0]
        t = ts[0]

        dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(self.dt))

        noise_shape = (y0.shape[0], y0.shape[-1])
        X = [y0]

        for n in range(1, T):
            a_t = actions[..., n - 1]

            X_old = X[n - 1]
            if noise.ndim == 3:
                X_new = (
                    X_old
                    + self.drift_func(X_old, a_t, t) * self.dt
                    + self.diff_func(X_old, a_t, t, noise[..., n - 1])
                    * dw_dist.sample(noise_shape).to(self.dt.device)
                )
            else:
                X_new = (
                    X_old
                    + self.drift_func(X_old, a_t, t) * self.dt
                    + self.diff_func(X_old, a_t, t, noise)
                    * dw_dist.sample(noise_shape).to(self.dt.device)
                )

            # New clip function manually
            X_new = self.clip(X_new)
            X.append(X_new)
            t = t + self.dt

        return torch.stack(X, dim=-1)


def train_sde(
        env_class=None,
        env_params=None,
        PROJECT="model-train-stoch-cartpole",
        NAME="nsde",
        F_S=4,
        F_A=1,
        dt_value: float=0.02,
        N_SAMPLES=60000,  # num of trajectories
        EPSILON=1,
        TRAJ_LEN=20,
        N_JOBS=-1,
        load_gen_f: MLP=None,
        load_gen_g: MLP=None,
        load_disc=None,
        mean_value=None,
        std_value=None,
        BATCH_SIZE=256,
        TRAIN_F=False,
        f_model_params=None,
        g_model_params=None,
        NOISE_TYPE="scalar",
        G_LR=8e-4,
        d_model_class=CollapseDisc,
        d_model_params=None,
        D_LR=8e-5,
        MAX_EPOCHS=8000,
        N_CRITIC=1,
        REG_TERM=10,
        LOG_FREQ=500,
        VERBOSE=False,
        LOG_WANDB=True,
        SAVE_MODEL=True,
        device="cuda",
):
    if f_model_params is None:
        f_model_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=100, num_layers=4)
    if g_model_params is None:
        g_model_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=32, num_layers=2)
    if d_model_params is None:
        d_model_params = dict(traj_len=TRAJ_LEN, num_layers=5)

    # Time steps
    dt = torch.tensor(dt_value, device=device)
    ts = torch.arange(TRAJ_LEN, device=device) * dt
    ts_ = ts.numpy(force=True)

    # Generate data (N, T, F_s + F_a)
    train_dataset = generate_dataset_parallel(
        env_class, env_params, n_samples=N_SAMPLES, epsilon=EPSILON,
        traj_min_len=TRAJ_LEN, n_jobs=N_JOBS
    )
    if mean_value is None or std_value is None:
        mean_value, std_value = get_mean_std(
            train_dataset, f_s=F_S, standardise_in_signal=True
        )
    train_dataset = standardise(train_dataset, mean_value, std_value)
    train_dataset = TrajDataset(train_dataset, f_s=F_S)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator(device=device))
    # Infinite train loader
    inf_trainiter = get_inf_iterloader(train_loader)

    ##############################
    # Eval data for visualization
    ##############################
    # Time indexes in a path to visualize the distribution
    vis_times = [3, 7, 11, 15, 19]
    assert max(vis_times) < TRAJ_LEN, f"Highest index is {max(vis_times)}"

    # Generate eval data for visualization (N, T, F_s + F_a)
    vis_data = generate_dataset_parallel(
        env_class, env_params, n_samples=200, epsilon=EPSILON,
        traj_min_len=TRAJ_LEN, n_jobs=N_JOBS
    )
    vis_data = standardise(vis_data, mean_value, std_value)
    vis_x0s = vis_data[:, 0, :F_S].to(device)  # N, F_s
    vis_actions = vis_data[..., F_S:].transpose(-1, -2).to(device)  # N, F_a, T
    vis_data_ = vis_data[..., :F_S].numpy(force=True).reshape(vis_data.shape[0], -1)  # N, F_s * T

    # Target trajectories of each feature to visualize
    # (N, F_s, T)
    vis_target = vis_data[..., :F_S].transpose(-1, -2).to(device)
    vis_feature_ = vis_target.numpy(force=True)

    ########################
    # Init generator
    ########################
    # SDE generator with a drift function f (learned from ODE)
    # and a diffusion function g

    # Drift function f
    if load_gen_f is None:
        f_model = MLP(**f_model_params)
    else:
        f_model = load_gen_f
        if TRAIN_F:
            linear_layers = [m for m in f_model._model if isinstance(m, nn.Linear)]
            print("Unfreezing last 1 layers")
            for layer in linear_layers[-1:]:
                layer.weight.requires_grad_(True)
                layer.bias.requires_grad_(True)

    # Diffusion function g
    if load_gen_g is None:
        g_model = MLP(**g_model_params)
    else:
        g_model = load_gen_g

    # SDE generator
    model_g = SDEBlock(
        SDEFunc(f_model=f_model, g_model=g_model),
        dt=dt,
        method='euler',
        noise_type=NOISE_TYPE,
    ).to(device)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=G_LR, betas=(0, 0.9))

    #########################
    # Init MLP discriminator
    #########################
    if load_disc is None:
        model_d = d_model_class(**d_model_params).to(device)
    else:
        model_d = load_disc
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=D_LR, betas=(0, 0.9))

    if LOG_WANDB:
        wandb.init(
            project=PROJECT,
            name=NAME,
            config={
                "N_SAMPLES": N_SAMPLES,
                "TRAJ_LEN": TRAJ_LEN,
                "MAX_EPOCHS": MAX_EPOCHS,
                "D_LR": D_LR,
                "G_LR": G_LR,
                "BATCH_SIZE": BATCH_SIZE,
                "REG_TERM": REG_TERM,
                "N_CRITIC": N_CRITIC,
                "DT": dt.item(),
                "f_model_params": f_model_params,
                "g_model_params": g_model_params,
                "d_model_params": d_model_params,
                "d_model_class": d_model_class,
                "train_f": TRAIN_F,
                "env_class": env_class,
                "env_params": env_params,
                "noise_type": NOISE_TYPE,
            },
        )

    with torch.no_grad():
        odefunc = ODEFunc(f_model)
        odeblock = ODEBlock(odefunc, dt=dt, method="euler").to(device).eval()
        # (N, F_s, T)
        pred_ = odeblock(ts, vis_actions, vis_x0s).numpy(force=True)
        # Real vs ODE-generated trajectories of each feature
        plot_real_gen_signals(
            vis_feature_, pred_, ts_, epoch=None, n_samples=100, verbose=VERBOSE
        )

    pbar = trange(MAX_EPOCHS, unit="epoch")
    for epoch in pbar:
        model_g.train()
        model_d.train()

        loss_acc = 0
        pbar.set_description(f"Epoch {epoch}")
        for _ in range(N_CRITIC):
            ########################
            # Train discriminator
            ########################
            # (B, F_s), (B, F_a, T), (B, F_s, T)
            y0, actions, target = next(inf_trainiter)
            y0 = y0.to(device)
            target = target.to(device)
            actions = actions.to(device)

            optimizer_d.zero_grad()
            real_score = model_d(target, actions)  # (B, )
            D_real_loss = -torch.mean(real_score)

            generated_samp = model_g(ts, actions, y0)  # (B, F_s, T)
            generated_score = model_d(generated_samp, actions)  # (B, )
            D_fake_loss = torch.mean(generated_score)

            grad_penalty = get_grad_penalty(
                target, generated_samp, model_d, in_signal=actions, reg_param=REG_TERM
            )
            # Minimizing the negative Wasserstein distance: gen_score - real_score
            D_loss = torch.mean(D_fake_loss + D_real_loss) + grad_penalty

            D_loss.backward()
            optimizer_d.step()

        ####################
        # Train generator
        ####################
        optimizer_g.zero_grad()
        gen_samp = model_g(ts, actions, y0)
        G_loss = -torch.mean(model_d(gen_samp, actions))
        G_loss.backward()
        optimizer_g.step()

        pbar.set_postfix(
            D_loss_train=D_loss.item(),
            Gen_loss_train=G_loss.item(),
            Grad_Pen=grad_penalty.item(),
        )

        if wandb.run is not None:
            wandb.log(
                {"train_D_loss": D_loss.item(),
                 "train_G_loss": G_loss.item(),
                 "train_grad_pen": grad_penalty.item(),
                 "W_dist_train": (grad_penalty - D_loss).item(),
                 "epoch": epoch}
            )

        ###################
        # Visualize eval
        ###################
        if epoch % LOG_FREQ == 0 or epoch + 1 == MAX_EPOCHS:
            with torch.no_grad():
                model_g.eval()
                model_d.eval()

                # (N, F_s, T)
                vis_samp_ = model_g(ts, vis_actions, vis_x0s).numpy(force=True)
                # Real vs SDE-generated trajectories of each feature
                plot_real_gen_signals(
                    vis_feature_, vis_samp_, ts_, epoch=epoch, n_samples=100, verbose=VERBOSE
                )

                # Real vs SDE-generated distribution of each feature at specified time points
                plot_cond_dist(
                    vis_times, vis_feature_, vis_samp_, epoch=epoch, verbose=VERBOSE
                )
                # Real vs SDE-generated vs ODE-generated distribution of each feature at specified time points
                vis_samp_f_ = odeblock(ts, vis_actions, vis_x0s).numpy(force=True)
                plot_cond_dist(
                    vis_times, vis_feature_, vis_samp_, vis_samp_f_, epoch=epoch, verbose=VERBOSE
                )

                # Real vs generated PCA representation of time for each feature
                plot_feature_latent(
                    vis_feature_, vis_samp_, epoch=epoch, tsne=False, verbose=VERBOSE
                )
                # Real vs generated TSNE representation of time for each feature
                plot_feature_latent(
                    vis_feature_, vis_samp_, epoch=epoch, tsne=True, verbose=VERBOSE
                )
                # Real vs generated PCA representation of features at specified time points
                plot_feat_latent_time_cond(
                    vis_times, vis_feature_, vis_samp_, epoch=epoch, tsne=False, verbose=VERBOSE,
                )
                # Real vs generated TSNE representation of features at specified time points
                plot_feat_latent_time_cond(
                    vis_times, vis_feature_, vis_samp_, epoch=epoch, tsne=True, verbose=VERBOSE,
                )

    if SAVE_MODEL:
        env = env_class(**env_params)
        if wandb.run is not None:
            save_model_gan(
                model_g, model_d, repr(env), run_id=wandb.run.id, mean_value=mean_value, std_value=std_value,
                **wandb.run.config
            )
        else:
            save_model_gan(
                model_g, model_d, repr(env), run_id=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                mean_value=mean_value, std_value=std_value,
            )


    wandb.finish()
    return model_g, mean_value, std_value

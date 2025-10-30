
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
import wandb
import math
import torch.nn as nn


def get_batch(dataset: torch.Tensor, nreps=1, traj_len=20, f_s=4):
    """Sample from dataset

    Args:
        dataset (torch.Tensor): Dataset to sample from. Shape: B, T, f_s + f_a
        nreps (int, optional): Number of repetitions of samples. Defaults to 1.
        traj_len (int, optional): Traj Len. Defaults to 20.
        f_s (int, optional): Number of states. Defaults to 4.

    Returns:
        s_0, a_T, target
    """
    # dataset: N, T, F_s + F_a
    # f_s: number of dimensions of the feature space
    # T = dataset.shape[1]
    idx = torch.arange(dataset.shape[0])
    idx = torch.tile(idx, (nreps, 1)).flatten()
    # Take sample:
    data_len = dataset.shape[1]
    traj_len = traj_len
    s_0 = torch.randint(0, max(data_len - traj_len, 1), size=idx.shape)

    sample = dataset[idx]
    sample = [sample[i, s_0[i]: s_0[i] + traj_len]
              for i in range(sample.shape[0])]
    sample = torch.stack(sample, dim=0)
    # Set to torch
    # sample = torch.from_numpy(sample)

    # Extract states, and targets
    target = sample[..., :f_s]
    s_0 = target[:, 0]
    a_T = sample[..., f_s:]

    # Take time as the last dimension
    target = target.transpose(-1, -2)
    a_T = a_T.transpose(-1, -2)

    # target: N, F_s, T
    # a_T: N, F_a, T

    assert s_0.shape[-1] == f_s, 'States shape is wrong'
    assert a_T.shape[1] == dataset.shape[-1] - f_s, 'Action shape is wrong'
    assert target.shape[1] == f_s, 'Action shape is wrong'
    return s_0, a_T, target


def calc_correlation(target_samp, out_samp, feature_idx=0):
    # target_samp/out_samp: (F_s, T)
    # correlation between i-th feature of target and out across T observations
    return torch.corrcoef(torch.stack([target_samp[feature_idx], out_samp[feature_idx]], dim=0))[0, -1].item()


def save_model(model, env_name, run_id=None, mean_value=None, std_value=None, **kwargs):
    dir = Path("../checkpoints")
    dir.mkdir(exist_ok=True)
    model_path = dir / f'{env_name}_ode_trans_{run_id}.pt'

    torch.save({'state_dict': model.state_dict(),
                'mean': mean_value,
                'std': std_value,
                **kwargs}, model_path)
    print(f'Model Saved! {model_path}')


def save_model_gan(model_g, model_d, env_name, run_id=None, mean_value=None, std_value=None, **kwargs):
    dir = Path("../checkpoints")
    dir.mkdir(exist_ok=True)
    model_path = dir / f'{env_name}_sde_trans_{run_id}.pt'

    torch.save({'state_dict_gen': model_g.state_dict(),
                'state_dict_disc': model_d.state_dict(),
                'mean': mean_value,
                'std': std_value,
                **kwargs}, model_path)
    print(f"Model Saved! {model_path}")


def save_model_ens(model, env_name, run_id=None, **kwargs):
    dir = Path("../checkpoints")
    dir.mkdir(exist_ok=True)
    model_path = dir / f'{env_name}_ensemble_trans_{run_id}.pt'

    # Handle EnsembleBlock objects specially
    if hasattr(model, 'ensemble_model'):
        torch.save({
            'state_dict': model.ensemble_model.state_dict(),
            'elite_model_idxes': model.elite_model_idxes,
            'scaler_mu': model.scaler.mu,
            'scaler_std': model.scaler.std,
            **kwargs
        }, model_path)
    else:
        # Original behavior for nn.Module objects
        torch.save({
            'state_dict': model.state_dict(),
            **kwargs
        }, model_path)

    print(f'Model Saved! {model_path}')


def plot_evaluation_plots(out, target, epoch, y0, title='', verbose=False, include_t=False):
    """Plot evaluation plots for CartPole

    Args:
        out (_type_): _description_
        target (_type_): _description_
        epoch (_type_): _description_
        y0 (_type_): _description_
        title (str, optional): _description_. Defaults to ''.
        verbose (bool, optional): _description_. Defaults to False.
        include_t (bool, optional): True if the out, target and y0 args contain time as their last dimension. Defaults to False.
    """
    start_title = title.strip() + ' ' if title.strip() != '' else ''
    _target = target.detach().cpu().transpose(-1, -2).numpy()
    _out = out.detach().cpu().transpose(-1, -2).numpy()
    _y0 = y0.detach().cpu().numpy()
    
    # _target, _out: B, T, F_s
    if include_t:
        _target = _target[..., :-1]
        _out = _out[..., :-1]
        _y0 = _y0[..., :-1]
    

    ####################################
    # Plot Transition of a function

    # Take the first batch
    _y0 = _y0[0]
    _target_samp = _target[0]
    _out_samp = _out[0]

    _target_samp_n_y0 = _target_samp
    _out_samp_n_y0 = _out_samp

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, layout='tight')
    plt.suptitle(f'{start_title}Epoch: {epoch:0>3d}')
    x_values = np.arange(0, _target_samp_n_y0.shape[0])
    ax1.plot(x_values, _target_samp_n_y0[:, 0], label='Target')
    ax1.plot(x_values, _out_samp_n_y0[:, 0], label='Prediction')
    ax1.plot(0, _y0[0], marker='x', color='black')
    ax1.plot(1, _target_samp_n_y0[1, 0], marker='x', color='red')
    ax1.plot(1, _out_samp_n_y0[1, 0], marker='x', color='red')
    ax1.set_title(f'{start_title}Position')
    ax1.legend()

    ax2.plot(x_values, _target_samp_n_y0[:, 1])
    ax2.plot(x_values, _out_samp_n_y0[:, 1])
    ax2.plot(0, _y0[1], marker='x', color='black')
    ax2.plot(1, _target_samp_n_y0[1, 1], marker='x', color='red')
    ax2.plot(1, _out_samp_n_y0[1, 1], marker='x', color='red')
    ax2.set_title(f'{start_title}Velocity')

    ax3.plot(x_values, _target_samp_n_y0[:, 2])
    ax3.plot(x_values, _out_samp_n_y0[:, 2])
    ax3.plot(0, _y0[2], marker='x', color='black')
    ax3.plot(1, _target_samp_n_y0[1, 2], marker='x', color='red')
    ax3.plot(1, _out_samp_n_y0[1, 2], marker='x', color='red')
    ax3.set_title(f'{start_title}Pole Angle')

    ax4.plot(x_values, _target_samp_n_y0[:, 3])
    ax4.plot(x_values, _out_samp_n_y0[:, 3])
    ax4.plot(1, _target_samp_n_y0[1, 3], marker='x', color='red')
    ax4.plot(1, _out_samp_n_y0[1, 3], marker='x', color='red')
    ax4.plot(0, _y0[3], marker='x', color='black')
    ax4.set_title(f'{start_title}Pole Angular\nVelocity')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, color='black', linestyle='dotted')

    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({f'{start_title}Transition': wandb.Image(fig)})
    # plt.show()
    ########################
    # Plot State Mean Error:

    # _target: [B, T, F]
    # _out: [B, T, F]
    diff = np.power(_target - _out, 2)

    fig, ax = plt.subplots()
    # plt.boxplot(diff, showsmean=True)
    ax.boxplot(diff.mean(axis=1), showmeans=True)
    ax.set_title(f'{start_title}MSE Across Time (Epoch: {epoch:0>3d})')
    ax.set_xticklabels(
        ['Position', 'Velocity', 'Pole Angle', 'Pole Angular\nVelocity'])
    if wandb.run is not None:
        wandb.log({f'{start_title}MSE (Time)': wandb.Image(fig)})
    #######################
    # Plot State/Time Error:

    mean_time_error = diff.mean(axis=0)
    std_time_error = diff.std(axis=0)

    # Plot
    fig, ax = plt.subplots()
    # plt.figure(figsize=(4,4))
    ticks = np.arange(mean_time_error.shape[0])
    ax.plot(ticks, mean_time_error[:, 0], label='Position')
    ax.fill_between(ticks, mean_time_error[:, 0] - std_time_error[:, 0],
                    mean_time_error[:, 0] + std_time_error[:, 0], alpha=0.3)

    ax.plot(ticks, mean_time_error[:, 1], label='Velocity')
    ax.fill_between(ticks, mean_time_error[:, 1] - std_time_error[:, 1],
                    mean_time_error[:, 1] + std_time_error[:, 1], alpha=0.3)

    ax.plot(ticks, mean_time_error[:, 2], label='Pole Angle')
    ax.fill_between(ticks, mean_time_error[:, 2] - std_time_error[:, 2],
                    mean_time_error[:, 2] + std_time_error[:, 2], alpha=0.3)

    ax.plot(ticks, mean_time_error[:, 3], label='Pole Angular Velocity')
    ax.fill_between(ticks, mean_time_error[:, 3] - std_time_error[:, 3],
                    mean_time_error[:, 3] + std_time_error[:, 3], alpha=0.3)
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('MSE')
    # ax.set_xlim(1)
    # ax.set_xticklabels(ticks)
    ax.set_title(f'{start_title}Epoch: {epoch:0>3d}')
    # fig.tight_layout()
    if wandb.run is not None:
        wandb.log({f'{start_title}Average Error per Step': wandb.Image(fig)})

    #########################
    # Correlation
    corr_lst = [np.array(list(map(lambda x: calc_correlation(
        x[0], x[-1], feature_idx=i), zip(target, out)))) for i in range(4)]
    corr_lst = np.array(corr_lst).T
    fig, ax = plt.subplots()
    ax.boxplot(corr_lst, showmeans=True)
    ax.set_title(f'{start_title}Correlation (Epoch: {epoch:0>3d})')
    ax.set_xticklabels(
        ['Position', 'Velocity', 'Pole Angle', 'Pole Angular\nVelocity'])
    if wandb.run is not None:
        wandb.log({f'{start_title}Correlation': wandb.Image(fig)})
    if verbose:
        plt.show()
    else:
        plt.close('all')

# Log feature loss


# @torch.no_grad()
# def log_feature_loss(out, target, epoch, val=False):
#     # feature_loss: [F]
#     feature_loss = F.mse_loss(out, target, reduction='none').mean(dim=(0, -1))
#     dict = {'epoch': epoch}
#     comment = '' if not val else 'Val '
#     feature_name = ['Position', 'Velocity',
#                     'Pole Angle', 'Pole Angular Velocity']
#     feature_name = [comment + feature for feature in feature_name]
#     for idx, name in zip(range(feature_loss.shape[0]), feature_name):
#         dict[name] = feature_loss[idx]
#     # print(dict)
#     if wandb.run is not None:
#         wandb.log(dict)
        
@torch.no_grad()
def log_feature_loss(out, target, epoch, val=False):
    # feature_loss: (F_s,)
    feature_loss = F.mse_loss(out, target, reduction="none").mean(dim=(0, -1))
    to_log = {"epoch": epoch}
    comment = "" if not val else "Val "
    # feature_name = ["Position", "Velocity", "Pole Angle", "Pole Angular Velocity"]
    feature_name = [comment + f'Feature: {i}' for i in range(feature_loss.shape[0])]
    for idx, name in zip(range(feature_loss.shape[0]), feature_name):
        to_log[name] = feature_loss[idx]
    # print(dict)
    if wandb.run is not None:
        wandb.log(to_log)


def log_outside_boundary(out, target, epoch, x_threshold=2.4, theta_threshold_radians=12 * 2 * math.pi / 360):
    # out: [B, F_s, T]
    # theta_threshold_radians=12 * 2 * math.pi / 360
    # x_threshold = 2.4
    def count_over_threshold(out):
        return (~(out[:, 0] < x_threshold) | ~(out[:, 0] > -x_threshold) | ~(out[:, 2] > -theta_threshold_radians) | ~(out[:, 2] < theta_threshold_radians)).sum()
    n_over_threshold = count_over_threshold(out) - count_over_threshold(target)
    # return count_over_threshold(out), count_over_threshold(target)
    if wandb.run is not None:
        wandb.log({'epoch': epoch, 'Val n_illegal': n_over_threshold})


def standardise(x, mean_y0, std_y0):
    return (x - mean_y0) / std_y0


def unstandardise(x, mean_y0, std_y0):
    return std_y0 * x + mean_y0


def plot_eval_figures_all_features(out, target, ts, n_samples=100, title='', epoch=0, verbose=False):
    # out/target: B, F_s, T
    out, target = out[:n_samples], target[:n_samples]  # Limit to 100 samples for plotting by default
    start_title = title.strip() + " " if title.strip() != "" else ""

    # T
    ts_ = ts.numpy(force=True)

    # F_s, T, B
    target_ = target.permute(1, -1, 0).numpy(force=True)
    out_ = out.permute(1, -1, 0).numpy(force=True)

    ##############
    # Plot Signals
    ##############
    for f_s in range(target_.shape[0]):
        fig, ax = plt.subplots(1, 1)
        ax.plot(ts_, target_[f_s], color="tab:blue", alpha=0.3, label='Real')
        ax.plot(ts_, out_[f_s], color="tab:red", alpha=0.3, label='Generated')
        ax.set_title(f"{start_title}Feature: {f_s} Epoch: {epoch:0>3d}")

        ax.set_xlabel("Time")
        # Both handles and identical labels for "Real" (and similarly for "Generated")
        handle, label = ax.get_legend_handles_labels()
        # Avoid repeating identical labels in the legend
        handle = [handle[0], handle[-1]]
        label = [label[0], label[-1]]
        plt.legend(handle, label, loc="best")

        if wandb.run is not None:
            wandb.log({f"{start_title}Feature Signal: {f_s}": wandb.Image(fig)})

    show_figures(verbose)

    ##############
    # Plot Error
    ##############
    error = (target_ - out_) ** 2
    mean_error = error.mean(axis=-1)
    std_error = error.std(axis=-1)
    for f_s in range(target_.shape[0]):
        fig, ax = plt.subplots(1, 1)
        feat_meanerror = mean_error[f_s]
        feat_stderror = std_error[f_s]
        ax.plot(ts_, feat_meanerror)
        ax.fill_between(
            ts_,
            feat_meanerror + feat_stderror,
            feat_meanerror - feat_stderror,
            alpha=0.2, edgecolor=None
        )
        ax.set_title(f"{start_title}MSE Feature: {f_s} Epoch: {epoch:0>3d}")
        ax.set_xlabel("Time")

        if wandb.run is not None:
            wandb.log({f"{start_title}MSE Feature: {f_s}": wandb.Image(fig)})
    
    show_figures(verbose)

    ##############
    # Correlation
    ##############
    # [(B,)] * F_s
    corr_lst = [
        np.array(
            list(
                map(
                    lambda x: calc_correlation(x[0], x[-1], feature_idx=i),
                    zip(target, out),
                )
            )
        )
        for i in range(target_.shape[0])
    ]
    for f_s in range(target_.shape[0]):
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(corr_lst[f_s], showmeans=True)
        ax.set_title(f"{start_title}Corr Feature: {f_s} Epoch: {epoch:0>3d}")

        if wandb.run is not None:
            wandb.log({f"{start_title}Corr Feature: {f_s}": wandb.Image(fig)})

    show_figures(verbose)


def show_figures(verbose):
    if verbose:
        plt.show()
    else:
        plt.close("all")
    

###################
# Model Structures
##################
class MLP(nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh=False, selu=False):
        super().__init__()
        if not selu:
            model = [nn.Linear(in_size, mlp_size),
                    nn.LeakyReLU(0.2, inplace=True)]
            for _ in range(num_layers - 1):
                model.append(nn.Linear(mlp_size, mlp_size))
                model.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            model = [nn.Linear(in_size, mlp_size),
                    nn.SELU()]
            for _ in range(num_layers - 1):
                model.append(nn.Linear(mlp_size, mlp_size))
                model.append(nn.SELU())
            
        model.append(nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)
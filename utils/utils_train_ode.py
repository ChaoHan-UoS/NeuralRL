import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
import wandb
# import fire

from utils.utils import (
    standardise, MLP, plot_eval_figures_all_features, log_feature_loss, save_model
)
from utils.utils_data import TrajDataset, generate_dataset_parallel, get_min_max_vel, get_mean_std
from utils.utils_models import ODEBlock, ODEFunc, ODEBlockManualClipping


def train_ode(
        env_class=None,
        env_params=None,
        project_name="model-train-stoch-cartpole",
        name="node",
        train_dataset: torch.Tensor=None,
        val_dataset: torch.Tensor=None,
        N_SAMPLES=100000,
        EPSILON=0.5,
        TRAJ_LEN=50,
        n_jobs=1,
        f_s=4,
        train_on_standardised=True,
        standardise_in_signal=True,
        load_block=None,
        f_model_params=None,
        clip_model=False,
        dt: float=0.02,
        method="euler",
        aug_dim=0,
        BATCH_SIZE=128,
        criterion=None,
        LR=8e-4,
        MAX_EPOCHS=100,
        log_freq=25,
        verbose=False,
        LOG_WANDB=True,
        SAVE_MODEL=True,
        device="cuda",
):
    if f_model_params is None:
        f_model_params = dict(in_size=5, out_size=4, mlp_size=100, num_layers=5)

    # Time steps
    dt = torch.tensor(dt, device=device).to(torch.float)
    ts = torch.arange(TRAJ_LEN, device=device) * dt

    # Get data from path
    if train_dataset is None:
        print("Generating training dataset")
        # Generate data (N, T, F_s + F_a)
        train_dataset = generate_dataset_parallel(
            env_class, env_params, n_samples=N_SAMPLES, epsilon=EPSILON,
            traj_min_len=TRAJ_LEN, n_jobs=n_jobs
        )
    else:
        N_SAMPLES = train_dataset.shape[0]
        print(f"N_SAMPLES set to {N_SAMPLES}")

    if val_dataset is None:
        print('Generating validation dataset')
        val_dataset = generate_dataset_parallel(
            env_class, env_params, n_samples=BATCH_SIZE * 50, epsilon=EPSILON,
            traj_min_len=TRAJ_LEN, n_jobs=n_jobs
        )
    else:
        print('Loaded validation set from file')

    if train_on_standardised:
        mean_value, std_value = get_mean_std(
            train_dataset, f_s=f_s, standardise_in_signal=standardise_in_signal
        )
    else:
        print('Training on unstandardised')
        mean_value = 0
        std_value = 1

    train_dataset = standardise(train_dataset, mean_value, std_value)
    if clip_model:
        min_lim_vel, max_lim_vel = get_min_max_vel(train_dataset)
        print("Clipped Model")
    val_dataset = standardise(val_dataset, mean_value, std_value)

    train_dataset = TrajDataset(train_dataset, f_s=f_s)
    val_dataset = TrajDataset(val_dataset, f_s=f_s)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            generator=torch.Generator(device=device))

    if load_block is None:
        # Init an ODE block
        f_model = MLP(**f_model_params)
        if clip_model:
            block = ODEBlockManualClipping(
                ODEFunc(f_model),
                dt=dt,
                method=method,
                min_lim_vel=min_lim_vel,
                max_lim_vel=max_lim_vel
            )
        else:
            block = ODEBlock(
                ODEFunc(f_model),
                dt=dt,
                method=method
            )
    else:
        # Use the loaded ODE block
        block = load_block
    block.to(device)

    criterion = nn.MSELoss() if criterion is None else criterion
    optimizer = torch.optim.Adam(block.parameters(), lr=LR)

    if LOG_WANDB:
        wandb.init(
            project=project_name,
            name=name,
            config={
                "lr": LR,
                "env_class": env_class,
                "env_params": env_params,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "train_size": len(train_dataset),
                "fixed_episode_length": TRAJ_LEN,
                "agent_type": "random" if EPSILON != 0 else "deterministic",
                "epsilon": EPSILON,
                "f_model_params": f_model_params,
                "criterion": criterion,
                "aug_dim": aug_dim,
                "method": method,
                "dt": dt,
                "standardise": train_on_standardised,
                "stand_in_signal": standardise_in_signal
            },
        )

    pbar = trange(MAX_EPOCHS, unit="epoch")
    for epoch in pbar:
        block.train()
        loss_acc = 0
        pbar.set_description(f"Epoch {epoch}")
        for batch_idx, (y0, actions, target) in enumerate(train_loader):
            y0 = y0.to(device)  # (B, F_s)
            target = target.to(device)  # (B, F_s, T)
            actions = actions.to(device)  # (B, F_a, T)
            optimizer.zero_grad()

            # Add augmented dimensions
            if aug_dim > 0:
                y0 = torch.concatenate(
                    [y0, torch.zeros((y0.shape[0], aug_dim)).to(y0)], dim=-1
                )

            out = block(ts, actions, y0)  # (B, F_s, T)

            # Remove augmented dimensions
            if aug_dim > 0:
                out = out[:, :f_s]

            loss = criterion(out, target)
            loss_acc += loss.item()

            loss.backward()
            optimizer.step()
        loss_acc /= len(train_loader)
        pbar.set_postfix(train_loss=loss_acc)
        log_feature_loss(out, target, epoch)
        if wandb.run is not None:
            wandb.log({"train_loss": loss_acc, "epoch": epoch})

        #################
        # Visualization
        ################
        if epoch % log_freq == 0 or epoch + 1 == MAX_EPOCHS:
            # Visualize training
            plot_eval_figures_all_features(
                out, target, ts, epoch=epoch, verbose=verbose
            )

            # Visualize validation
            block.eval()
            val_loss_acc = 0
            with torch.no_grad():
                for batch_idx, (y0, actions, target) in enumerate(val_loader):
                    y0 = y0.to(device)
                    target = target.to(device)
                    actions = actions.to(device)

                    # Add augmented dimension
                    if aug_dim > 0:
                        y0 = torch.concatenate(
                            [y0, torch.zeros((y0.shape[0], aug_dim)).to(y0)], dim=-1
                        )

                    out = block(ts, actions, y0)

                    # Remove augmented dimension
                    if aug_dim > 0:
                        out = out[:, :f_s]

                    loss = criterion(out, target)
                    val_loss_acc += loss.item()
                val_loss_acc /= len(val_loader)
            pbar.set_postfix(train_loss=loss_acc, val_loss=val_loss_acc)
            log_feature_loss(out, target, epoch, val=True)
            # log_outside_boundary(out, target, epoch)
            if wandb.run is not None:
                wandb.log({"val_loss": val_loss_acc, "epoch": epoch})

            plot_eval_figures_all_features(
                out, target, ts, epoch=epoch, verbose=verbose, title="Val"
            )

    ################
    # Saving Model
    ################
    if SAVE_MODEL:
        env = env_class(**env_params)

        run_id = wandb.run.id if wandb.run is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        to_add = dict()
        if clip_model:
            to_add = dict(min_lim_vel=min_lim_vel, max_lim_vel=max_lim_vel, clip_model=clip_model)
        to_add["f_model_params"] = f_model_params
        to_add["dt"] = dt
        to_add["method"] = method

        save_model(
            block, repr(env), run_id=run_id, mean_value=mean_value, std_value=std_value, **to_add
        )

    wandb.finish()
    return block, mean_value, std_value


# if __name__ == "__main__":
#     wandb.login()
#     fire.Fire(train_ode)

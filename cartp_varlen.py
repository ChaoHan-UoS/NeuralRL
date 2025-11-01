import random
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from env.cartpole import (
    CartPoleEnv, CartPoleEnvVaryingLength,
    CartPoleEnvStochCon, CartPoleEnvStochVaryingLength
)
from sac.sac import SAC
from utils.utils_train_sac import eval_agent, eval_agent_invdyn
from utils.utils import MLP
from utils.utils_models import (
    LearnedTrans, LearnedTransDiff, ODEFunc, ODEBlock, SDEFunc, SDEBlock,
    AugFModel, AugODEFunc, InverseDynamicsModel
)
from utils.utils_train_ode import train_ode
from utils.utils_train_sde import CollapseDisc, train_sde
from utils.utils_train_ensemble import train_ensemble
from utils.mpc_utils import ReplayMemory
from utils.running_stats_utils import RunningStats
from mbpo.model import EnsembleBlock


# Env config
ENV_DICT = {
    'det': (CartPoleEnv, CartPoleEnvVaryingLength),
    'stoch': (CartPoleEnvStochCon, CartPoleEnvStochVaryingLength)
}
ENV_PARAMS = {
    'det': dict(verbose=False, discrete=False),
    'stoch': dict(verbose=False, force_mag=10, diff_sigma=2.0)
}

parser = ArgumentParser(description="CartPole policy adaptation with varying pole length")
parser.add_argument('--env', choices=ENV_DICT.keys(), default='stoch',
                    help="Environment type: 'stoch' (default) or 'det'")
parser.add_argument('--length', type=float, default=3.0,
                    help="Target pole length (default: 3.0)")
parser.add_argument('--model', choices=('ode', 'sde', 'ens'), default='sde',
                    help="Model type: 'ode', 'sde' (default), or 'ens'")

# Loading and logging
parser.add_argument('--policy-ckpt', type=str, required=True,
                    help="Source policy checkpoint run ID")
parser.add_argument('--model-ckpt', type=str, default="",
                    help="Source model checkpoint run ID")
parser.add_argument('--no-wandb', action='store_true', default=False,
                    help="Disable wandb logging")

# Target model training
parser.add_argument('--traj-len', type=int, default=20,
                    help="Trajectory length (default: 20)")
parser.add_argument('--model-batch-size', type=int, default=256,
                    help="Batch size for model training (default: 256)")
parser.add_argument('--model-lr', type=float, default=8e-4,
                    help="Learning rate for model training (default: 8e-4)")
parser.add_argument('--n-samples', type=int, default=100,
                    help="Number of trajectories for target model training (default: 100)")
parser.add_argument('--max-epochs-ode', type=int, default=500,
                    help="Maximum epochs for ODE model training (default: 500)")
parser.add_argument('--max-epochs-sde', type=int, default=8000,
                    help="Maximum epochs for SDE model training (default: 8000)")

# Inverse dynamics training
parser.add_argument('--num-steps', type=int, default=10_000,
                    help="Number of environment steps for inverse dynamics training (default: 10_000)")
parser.add_argument('--updates-per-step', type=int, default=1,
                    help="Gradient updates per environment step (default: 1)")
parser.add_argument('--batch-size', type=int, default=256,
                    help="Batch size (default: 256)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
parser.add_argument('--loss-weight', type=float, default=500,
                    help="Loss weight for inverse dynamics (default: 500)")
parser.add_argument('--replay-size', type=int, default=20_000,
                    help="Replay buffer size (default: 20_000)")

# Inverse dynamics model
parser.add_argument('--no-rms', dest='rms', action='store_false',
                    help="Disable running mean/std normalization")
parser.add_argument('--use-noise', action='store_true', default=False,
                    help="Add exploration noise to actions")
parser.add_argument('--noise-start', type=float, default=0.2,
                    help="Starting noise scale (default: 0.2)")
parser.add_argument('--noise-end', type=float, default=0.05,
                    help="Ending noise scale (default: 0.05)")
parser.add_argument('--prob-noise', type=float, default=0.2,
                    help="Probability of adding noise (default: 0.2)")
parser.add_argument('--use-delta-a', action='store_true', default=False,
                    help="Use delta action instead of absolute action")

# Target policy evaluation
parser.add_argument('--n-eval-episodes', type=int, default=1000,
                    help="Number of evaluation episodes (default: 1000)")


def main(args=None):
    if args is None:
        args = parser.parse_args()

    wandb.login()
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    LOG_WANDB = not args.no_wandb

    print(f"Device: {device}")
    print(f"Env: {args.env}")
    print(f"Model: {args.model}")
    print(f"Target pole length: {args.length}")
    print(f"Source Policy checkpoint: {args.policy_ckpt}")
    print(f"Source {args.model} model checkpoint: {args.model_ckpt}")

    ###########################
    # Source and Target Envs
    ###########################
    env_src_class, env_tar_class = ENV_DICT[args.env]
    env_src_params = ENV_PARAMS[args.env].copy()
    env_src = env_src_class(**env_src_params)
    F_S, F_A = 4, 1

    env_tar_params = env_src_params.copy()
    env_tar_params['pole_len'] = args.length
    env_tar = env_tar_class(**env_tar_params)

    ###########################
    # Source Policy
    ###########################
    gamma = 0.99
    tau = 0.005
    policy = "Gaussian"
    target_update_interval = 1
    auto_entropy_tuning = False
    lr_agent = 1e-3
    alpha = 0.2
    hidden_size = 200

    agent_src = SAC(
        num_inputs=env_src.observation_space.shape[0],
        action_space=env_src.action_space,
        gamma=gamma,
        tau=tau,
        policy=policy,
        target_update_interval=target_update_interval,
        automatic_entropy_tuning=auto_entropy_tuning,
        cuda=torch.cuda.is_available(),
        lr=lr_agent,
        alpha=alpha,
        hidden_size=hidden_size,
    )

    print(f'\nLoading source policy {args.policy_ckpt}')
    agent_src.load_checkpoint(f'checkpoints/{repr(env_src)}_mf_{args.policy_ckpt}.pt')

    if LOG_WANDB:
        wandb.init(
            project=f'rl-{repr(env_tar)}-adaptation',
            name=f'nonadapt-{args.length}',
            config={
                'env_config': dict(
                    env_src_class=env_src_class.__name__,
                    env_src_params=env_src_params,
                    env_tar_class=env_tar_class.__name__,
                    env_tar_params=env_tar_params,
                ),
                'sac_agent_src_params': dict(
                    num_inputs=F_S,
                    action_space=env_src.action_space,
                    gamma=gamma,
                    tau=tau,
                    policy=policy,
                    target_update_interval=target_update_interval,
                    automatic_entropy_tuning=auto_entropy_tuning,
                    cuda=torch.cuda.is_available(),
                    lr=lr_agent,
                    alpha=alpha,
                    hidden_size=hidden_size
                ),
                'category': 'nonadapt',
                'pole_length': args.length,
                'policy_ckpt': args.policy_ckpt,
            }
        )

    print('\nEvaluating non-adapted policy')
    all_results = dict()
    eval_rewards = eval_agent(agent_src, env_tar, args.n_eval_episodes, determ=True)[1]
    all_results['nonadapt_policy'] = eval_rewards
    all_results['length'] = args.length
    eval_rewards_mean = np.mean(eval_rewards)
    eval_rewards_std = np.std(eval_rewards)
    print(f"Non-adapted: {eval_rewards_mean:.2f} ± {eval_rewards_std:.2f}")

    if wandb.run is not None:
        wandb.log({"eval_rewards_mean": eval_rewards_mean,
                   "eval_rewards_std": eval_rewards_std})

    wandb.finish()

    ###########################
    # Source Transition Model
    ###########################
    print(f'\nLoading source {args.model} model {args.model_ckpt}')
    if args.model == 'ode':
        ode_trans_path = Path(f'checkpoints/{repr(env_src)}_ode_trans_{args.model_ckpt}.pt')
        ode_trans_ckpts = torch.load(ode_trans_path)

        f_model = MLP(**ode_trans_ckpts['f_model_params'])
        odefunc = ODEFunc(f_model)
        odeblock = ODEBlock(
            odefunc, dt=ode_trans_ckpts['dt'], method=ode_trans_ckpts['method']
        ).eval()
        odeblock.load_state_dict(ode_trans_ckpts['state_dict'])

        # Source ODE transition
        trans_src = LearnedTrans(
            odeblock, ode_trans_ckpts['mean'], ode_trans_ckpts['std'], f_s=F_S
        ).to(device)
    elif args.model == 'sde':
        sde_trans_path = Path(f'checkpoints/{repr(env_src)}_sde_trans_{args.model_ckpt}.pt')
        sde_trans_ckpts = torch.load(sde_trans_path)

        # Load a discriminator
        d_model_params = sde_trans_ckpts['d_model_params']
        model_d = CollapseDisc(**d_model_params).to(device)
        model_d.requires_grad_(True)

        # Load drift f and diffusion g of a generator
        f_model_params = sde_trans_ckpts['f_model_params']
        f_model = MLP(**f_model_params)
        g_model_params = sde_trans_ckpts['g_model_params']
        g_model = MLP(**g_model_params)
        dt_value = sde_trans_ckpts['DT']
        dt = torch.tensor(dt_value)
        model_g = SDEBlock(
            SDEFunc(f_model=f_model, g_model=g_model),
            dt=dt,
            method='euler',
            noise_type=sde_trans_ckpts['noise_type'],
        ).eval()
        model_g.load_state_dict(sde_trans_ckpts['state_dict_gen'])
        odeblock = ODEBlock(
            ODEFunc(model_g.func.f_model),
            dt=dt,
            method='euler'
        ).eval()

        # Source SDE drift transition
        trans_src = LearnedTrans(
            odeblock, sde_trans_ckpts['mean'], sde_trans_ckpts['std'], f_s=F_S
        ).to(device)
    elif args.model == 'ens':
        ens_trans_path = Path(f'checkpoints/{repr(env_src)}_ensemble_trans_{args.model_ckpt}.pt')
        ens_trans_ckpts = torch.load(ens_trans_path)

        ens_params = ens_trans_ckpts['ens_params']
        ensblock = EnsembleBlock(**ens_params)
        ensblock.ensemble_model.load_state_dict(ens_trans_ckpts['state_dict'])
        ensblock.elite_model_idxes = ens_trans_ckpts['elite_model_idxes']
        ensblock.scaler.mu = ens_trans_ckpts['scaler_mu']
        ensblock.scaler.std = ens_trans_ckpts['scaler_std']
        # Source ens transition
        trans_src = ensblock

    ###########################
    # Target Transition Model
    ###########################
    print(f'\nTraining target {args.model} model')
    project_name = f"model-train-{repr(env_tar)}"
    run_name = f"aug-{args.model}-{args.length}"
    if args.model == 'ode':
        # Augment ODE with trainable remap function
        odeblock.func.requires_grad_(False)
        remap_func_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=64, num_layers=1)
        aug_odefunc = AugODEFunc(
            odeblock.func.f_model, MLP(**remap_func_params), F_A
        )
        aug_odeblock = ODEBlock(
            aug_odefunc, dt=ode_trans_ckpts['dt'], method=ode_trans_ckpts['method']
        ).to(device)

        odeblock_tar, mean_tar, std_tar = train_ode(
            env_class=env_tar_class,
            env_params=env_tar_params,
            project_name=project_name,
            name=run_name,
            N_SAMPLES=args.n_samples,
            EPSILON=1,
            TRAJ_LEN=args.traj_len,
            n_jobs=-1,
            f_s=F_S,
            load_block=aug_odeblock,
            dt=0.02,
            method="euler",
            BATCH_SIZE=args.model_batch_size,
            LR=args.model_lr,
            MAX_EPOCHS=args.max_epochs_ode,
            log_freq=25,
            clip_model=False,
            LOG_WANDB=LOG_WANDB,
            SAVE_MODEL=True,
            device=device,
        )

        # Target ODE transition
        trans_tar = LearnedTransDiff(odeblock_tar, mean_tar, std_tar, f_s=F_S).to(device)
    elif args.model == 'sde':
        model_g.func.requires_grad_(False)
        remap_func_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=64, num_layers=1)
        aug_f_model = AugFModel(
            model_g.func.f_model, MLP(**remap_func_params), F_A
        )

        model_g_tar, mean_sde_tar, std_sde_tar = train_sde(
            env_class=env_tar_class,
            env_params=env_tar_params,
            PROJECT=project_name,
            NAME=run_name,
            F_S=F_S,
            F_A=F_A,
            dt_value=0.02,
            N_SAMPLES=args.n_samples,
            EPSILON=1,
            TRAJ_LEN=args.traj_len,
            N_JOBS=-1,
            load_gen_f=aug_f_model,
            load_gen_g=model_g.func.g_model,
            load_disc=model_d,
            BATCH_SIZE=args.model_batch_size,
            TRAIN_F=False,
            f_model_params=f_model_params,
            g_model_params=g_model_params,
            NOISE_TYPE="scalar",
            G_LR=args.model_lr,
            d_model_class=CollapseDisc,
            d_model_params=d_model_params,
            D_LR=args.model_lr * 0.1,
            MAX_EPOCHS=args.max_epochs_sde,
            N_CRITIC=1,
            REG_TERM=10,
            LOG_FREQ=500,
            VERBOSE=False,
            LOG_WANDB=LOG_WANDB,
            SAVE_MODEL=True,
            device=device,
        )

        # Target SDE drift transition
        odeblock_sde_tar = ODEBlock(
            ODEFunc(model_g_tar.func.f_model),
            dt=dt,
            method='euler'
        ).eval()
        trans_tar = LearnedTransDiff(
            odeblock_sde_tar, mean_sde_tar, std_sde_tar, f_s=F_S
        ).to(device)
    elif args.model == 'ens':
        ensblock_tar = EnsembleBlock(**ens_params, differ_trans=True)
        ensblock_tar.ensemble_model.load_state_dict(ens_trans_ckpts['state_dict'])

        # Freeze all but last layer
        ensblock_tar.ensemble_model.requires_grad_(False)
        ensblock_tar.ensemble_model.nn5.requires_grad_(True)

        ensblock_tar = train_ensemble(
            env_class=env_tar_class,
            env_params=env_tar_params,
            ens_params=ens_params,
            project_name=project_name,
            name=run_name,
            N_SAMPLES=args.n_samples * args.traj_len,
            load_block=ensblock_tar,
            BATCH_SIZE=args.model_batch_size,
            LR=args.model_lr,
            LOG_WANDB=LOG_WANDB,
            SAVE_MODEL=True,
            device=device,
        )
        # Target ens transition
        trans_tar = ensblock_tar

    ###########################
    # Inverse Dynamics Model
    ###########################
    print(f'\nTraining inverse dynamics model')
    a_low, a_high = env_tar.action_space.low, env_tar.action_space.high

    # Running mean/std normalization
    if args.rms:
        rms_s_tar = RunningStats(device, F_S)
        rms_s_next_src = RunningStats(device, F_S)

    # Init inverse model
    inv_model = InverseDynamicsModel(
        state_dim=F_S,
        action_dim=F_A,
        hidden_sizes=256,
        num_layers=2,
        use_delta_a=args.use_delta_a
    ).to(device)
    optimizer = optim.Adam(inv_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Init replay buffer
    TransInv = namedtuple(
        'TransInv',
        ('s_tar', 'a_src', 's_next_src') if args.use_delta_a else ('s_tar', 's_next_src')
    )
    memory_trans_inv = ReplayMemory(args.replay_size, TransInv)

    if LOG_WANDB:
        category = f'adapt-{args.model}'
        wandb.init(
            project=f'rl-{repr(env_tar)}-adaptation',
            name=f'{category}-{args.length}',
            config={
                'env_config': dict(
                    env_src_class=env_src_class.__name__,
                    env_src_params=env_src_params,
                    env_tar_class=env_tar_class.__name__,
                    env_tar_params=env_tar_params,
                ),
                'train_invdyn_params': dict(
                    num_env_steps=args.num_steps,
                    updates_per_step=args.updates_per_step,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    loss_weight=args.loss_weight,
                    rms=args.rms,
                    use_noise=args.use_noise,
                    noise_start=args.noise_start,
                    noise_end=args.noise_end,
                    prob_noise=args.prob_noise,
                    use_delta_a=args.use_delta_a,
                ),
                'category': category,
                'pole_length': args.length,
                'policy_ckpt': args.policy_ckpt,
                'model_ckpt': args.model_ckpt,
            }
        )

    # Training loop
    s_tar = env_tar.reset()[0]
    for step in range(1, args.num_steps + 1):
        # --- One target transition collection ---
        with torch.no_grad():
            # 1) Get source action and predict next sim state
            a_src = agent_src.select_action(s_tar, evaluate=True)
            s_next_src = trans_src(s_tar, a_src)

            # 2) Compute target action to reach next sim state
            a_src_ts = torch.from_numpy(a_src).float().to(device)
            s_tar_ts = torch.from_numpy(s_tar).float().to(device)
            s_next_src_ts = torch.from_numpy(s_next_src).float().to(device)

            if args.rms:
                rms_s_tar.push(s_tar_ts)
                rms_s_next_src.push(s_next_src_ts)

                # Normalize inputs
                s_tar_in = s_tar_ts.unsqueeze(0) if step == 1 \
                    else rms_s_tar.normalize(s_tar_ts).unsqueeze(0)
                s_next_src_in = s_next_src_ts.unsqueeze(0) if step == 1 \
                    else rms_s_next_src.normalize(s_next_src_ts).unsqueeze(0)
            else:
                s_tar_in = s_tar_ts.unsqueeze(0)
                s_next_src_in = s_next_src_ts.unsqueeze(0)

            if args.use_delta_a:
                delta_a_ts = inv_model(s_tar_in, s_next_src_in)
                delta_a = delta_a_ts.cpu().numpy().squeeze(0)
                a_tar = a_src + delta_a
            else:
                a_tar_ts = inv_model(s_tar_in, s_next_src_in)
                a_tar = a_tar_ts.cpu().numpy().squeeze(0)
            a_tar = np.clip(a_tar, a_low, a_high)

            if args.use_noise and random.random() < args.prob_noise:
                # Linear decay over num_steps
                noise_scale = args.noise_start + (args.noise_end - args.noise_start) * (step / args.num_steps)
                a_tar += np.random.randn(*a_tar.shape).astype(np.float32) * noise_scale
                a_tar = np.clip(a_tar, a_low, a_high)

        # 3) Apply target action in real env
        s_next_tar, _, terminated, truncated, _ = env_tar.step(a_tar)
        s_tar = s_next_tar if not terminated and not truncated else env_tar.reset()[0]

        # Store in buffer
        trans_inv = (s_tar_ts, a_src_ts, s_next_src_ts) if args.use_delta_a else (s_tar_ts, s_next_src_ts)
        memory_trans_inv.push(*trans_inv)

        # --- Supervised training of inverse model ---
        if len(memory_trans_inv) > args.batch_size:
            inv_model.train()
            for _ in range(args.updates_per_step):
                trans_inv_batch = memory_trans_inv.sample_tensor(args.batch_size)
                if args.use_delta_a:
                    s_tar_batch, a_src_batch, s_next_src_batch = trans_inv_batch
                else:
                    s_tar_batch, s_next_src_batch = trans_inv_batch

                if args.rms:
                    s_tar_in = rms_s_tar.normalize(s_tar_batch)
                    s_next_src_in = rms_s_next_src.normalize(s_next_src_batch)
                else:
                    s_tar_in = s_tar_batch
                    s_next_src_in = s_next_src_batch

                if args.use_delta_a:
                    delta_a_batch = inv_model(s_tar_in, s_next_src_in)
                    a_tar_batch = a_src_batch + delta_a_batch
                else:
                    a_tar_batch = inv_model(s_tar_in, s_next_src_in)

                s_next_tar_batch = trans_tar(s_tar_batch, a_tar_batch)
                loss = args.loss_weight * criterion(s_next_tar_batch, s_next_src_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % 500 == 0:
                print(f"Step {step}/{args.num_steps}, Loss: {loss.item():.5f}")
            if wandb.run is not None:
                wandb.log({'env_step': step, 'loss': loss.item()})

    ###########################
    # Final Evaluation
    ###########################
    print('\nEvaluating adapted policy')
    eval_rewards = eval_agent_invdyn(
        agent_src, env_tar, trans_src, inv_model, device, args.n_eval_episodes, determ=True,
        use_delta_a=args.use_delta_a, rms=args.rms,
        rms_s_tar=rms_s_tar if args.rms else None,
        rms_s_next_src=rms_s_next_src if args.rms else None
    )[1]

    key = f'adapt_{args.model.replace("-", "_")}_policy'
    all_results[key] = eval_rewards
    eval_rewards_mean = np.mean(eval_rewards)
    eval_rewards_std = np.std(eval_rewards)
    print(f"Adapted ({args.model}): {eval_rewards_mean:.2f} ± {eval_rewards_std:.2f}")

    if wandb.run is not None:
        wandb.log({"eval_rewards_mean": eval_rewards_mean,
                   "eval_rewards_std": eval_rewards_std})

    # Save results
    dir = Path(f'results_adaptation/{repr(env_tar)}')
    dir.mkdir(exist_ok=True, parents=True)
    file_name = f'results_{args.policy_ckpt}_{args.length}_tge{args.model}.pt'
    torch.save(all_results, dir / file_name)
    print(f"\nSaved results to {dir / file_name}")
    print('Done!')

    wandb.finish()


if __name__ == "__main__":
    main()

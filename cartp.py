from argparse import ArgumentParser
from pathlib import Path
import datetime

import numpy as np
import torch
import wandb

from env.cartpole import CartPoleEnv, CartPoleEnvStochCon
from utils.utils import MLP
from utils.utils_models import LearnedTrans, ODEFunc, ODEBlock, SDEFunc, SDEBlock
from utils.utils_train_ode import train_ode
from utils.utils_train_sde import CollapseDisc, train_sde
from utils.utils_train_ensemble import train_ensemble
from sac.sac import SAC
from utils.utils_train_sac import train_agent, eval_agent


# Env config
ENV_DICT = {'stoch': CartPoleEnvStochCon, 'det': CartPoleEnv}
ENV_PARAMS = {
    'stoch': dict(verbose=False, force_mag=20, diff_sigma=12),
    'det': dict(verbose=False, discrete=False)
}

parser = ArgumentParser(description="CartPole policy learning with model-based and model-free baselines")
parser.add_argument('--env', choices=ENV_DICT.keys(), default='stoch',
                    help="Environment type: 'stoch' (default) or 'det'")
parser.add_argument('--model', choices=('ode', 'sde', 'ens', 'mf'), default='sde',
                    help="Model type: 'ode', 'sde' (default), 'ens' or 'mf'")
parser.add_argument('--no-train-rl', dest='train_rl', action='store_false',
                    help="Skip RL agent training")

# Loading
parser.add_argument('--load-model', action='store_true', default=False,
                    help="Load pre-trained model instead of training")
parser.add_argument('--ode-ckpt', type=str, default="",
                    help="ODE model checkpoint run ID")
parser.add_argument('--sde-ckpt', type=str, default="",
                    help="SDE model checkpoint run ID")
parser.add_argument('--ens-ckpt', type=str, default="",
                    help="Ensemble model checkpoint run ID")

# Logging
parser.add_argument('--no-wandb', action='store_true', default=False,
                    help="Disable wandb logging")
parser.add_argument('--no-save-model', dest='save_model', action='store_false',
                    help="Do not save trained models")
parser.add_argument('--no-save-agent', dest='save_agent', action='store_false',
                    help="Do not save trained models")

# Model training
parser.add_argument('--traj-len', type=int, default=20,
                    help="Trajectory length (default: 20)")
parser.add_argument('--model-batch-size', type=int, default=256,
                    help="Batch size for model training (default: 256)")
parser.add_argument('--model-lr', type=float, default=8e-4,
                    help="Learning rate for model training (default: 8e-4)")
parser.add_argument('--n-samples-ode', type=int, default=25600,
                    help="Number of samples for ode model training (default: 25600)")
parser.add_argument('--max-epochs-ode', type=int, default=500,
                    help="Maximum epochs for ode model training (default: 500)")
parser.add_argument('--n-samples-sde', type=int, default=60_000,
                    help="Number of trajectory samples for sde model training (default: 60_000)")
parser.add_argument('--max-epochs-sde', type=int, default=8000,
                    help="Maximum epochs for ode model training (default: 8000)")

# Agent training
parser.add_argument('--replay-size', type=int, default=300_000,
                    help="Replay buffer size (default: 300_000)")
parser.add_argument('--batch-size', type=int, default=32,
                    help="Batch size for RL training (default: 32)")
parser.add_argument('--num-steps', type=int, default=500,
                    help="Number of training episodes (default: 500)")
parser.add_argument('--max-episode-len', type=int, default=500,
                    help="Maximum episode length (default: 500)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate for RL agent (default: 1e-3)")
parser.add_argument('--alpha', type=float, default=0.3,
                    help="SAC alpha parameter (default: 0.3)")
parser.add_argument('--hidden-size', type=int, default=200,
                    help="Hidden layer size for RL networks (default: 200)")


def main(args=None):
    if args is None:
        args = parser.parse_args()
    assert not (args.model == 'mf' and not args.train_rl), "Model-free approach requires RL training"

    wandb.login()
    LOG_WANDB = not args.no_wandb
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    env_class = ENV_DICT[args.env]
    env_params = ENV_PARAMS[args.env]
    env = env_class(**env_params)
    F_S, F_A = 4, 1

    print(f"Env: {args.env}")
    print(f"Model: {args.model}")
    print(f"Load model: {args.load_model}")
    print(f"Train RL: {args.train_rl}")
    print(f"Device: {device}")

    #########################
    # Model Training/Loading
    #########################
    project_name = f"model-train-{repr(args.env)}"
    if args.model == 'ode':
        if args.load_model and args.ode_ckpt:
            print(f'\nLoading ODE model {args.ode_ckpt}')
            ode_trans_path = Path(f'checkpoints/{repr(env)}_ode_trans_{args.ode_ckpt}.pt')
            ode_trans_ckpts = torch.load(ode_trans_path)

            f_model_params = ode_trans_ckpts['f_model_params']
            f_model = MLP(**f_model_params)
            odefunc = ODEFunc(f_model)
            odeblock = ODEBlock(odefunc, dt=ode_trans_ckpts['dt'], method=ode_trans_ckpts['method']).to(device)
            odeblock.load_state_dict(ode_trans_ckpts['state_dict'])
            mean_value = ode_trans_ckpts['mean']
            std_value = ode_trans_ckpts['std']
        else:
            print('\nTraining ODE model')
            f_model_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=100, num_layers=4)
            odeblock, mean_value, std_value = train_ode(
                env_class=env_class,
                env_params=env_params,
                project_name=project_name,
                name=args.model,
                N_SAMPLES=args.n_samples_ode,
                EPSILON=1,                      # random policy for data collection
                TRAJ_LEN=args.traj_len,
                n_jobs=-1,                      # -1 for all available CPU cores
                f_s=F_S,
                f_model_params=f_model_params,
                dt=0.02,
                method="euler",
                BATCH_SIZE=args.model_batch_size,
                LR=args.model_lr,
                MAX_EPOCHS=args.max_epochs_ode,
                log_freq=50,
                clip_model=False,
                LOG_WANDB=LOG_WANDB,
                SAVE_MODEL=args.save_model,
                device=device,
            )
        learned_trans = LearnedTrans(odeblock, mean_value, std_value, f_s=F_S)
    elif args.model == 'sde':
        # Load or train ODE first
        if args.load_model and args.ode_ckpt:
            print(f'\nLoading ODE model {args.ode_ckpt}')
            ode_trans_path = Path(f'checkpoints/{repr(env)}_ode_trans_{args.ode_ckpt}.pt')
            ode_trans_ckpts = torch.load(ode_trans_path)

            f_model_params = ode_trans_ckpts['f_model_params']
            f_model = MLP(**f_model_params)
            odefunc = ODEFunc(f_model)
            odeblock = ODEBlock(odefunc, dt=ode_trans_ckpts['dt'], method=ode_trans_ckpts['method']).to(device)
            odeblock.load_state_dict(ode_trans_ckpts['state_dict'])
            mean_value = ode_trans_ckpts['mean']
            std_value = ode_trans_ckpts['std']
        else:
            print('\nTraining ODE model')
            f_model_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=100, num_layers=4)
            odeblock, mean_value, std_value = train_ode(
                env_class=env_class,
                env_params=env_params,
                project_name=project_name,
                name="ode_for_sde",
                N_SAMPLES=args.n_samples_ode,
                EPSILON=1,
                TRAJ_LEN=args.traj_len,
                n_jobs=-1,
                f_s=F_S,
                f_model_params=f_model_params,
                dt=0.02,
                method="euler",
                BATCH_SIZE=args.model_batch_size,
                LR=args.model_lr,
                MAX_EPOCHS=args.max_epochs_ode,
                log_freq=50,
                clip_model=False,
                LOG_WANDB=LOG_WANDB,
                SAVE_MODEL=args.save_model,
                device=device,
            )

        # Load or train SDE
        if args.load_model and args.sde_ckpt:
            print(f'\nLoading SDE model {args.sde_ckpt}')
            sde_trans_path = Path(f'checkpoints/{repr(env)}_sde_trans_{args.sde_ckpt}.pt')
            sde_trans_ckpts = torch.load(sde_trans_path)

            # Load generator
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
            ).to(device)
            model_g.load_state_dict(sde_trans_ckpts['state_dict_gen'])
        else:
            print('\nTraining SDE model')
            # Use the frozen drift function f from ODE
            f_model = odeblock.get_f_model()
            f_model.requires_grad_(False)
            g_model_params = dict(in_size=F_S + F_A, out_size=F_S, mlp_size=32, num_layers=2)
            d_model_params = dict(traj_len=args.traj_len, num_layers=5)

            model_g = train_sde(
                env_class=env_class,
                env_params=env_params,
                PROJECT=project_name,
                NAME=args.model,
                F_S=F_S,
                F_A=F_A,
                dt_value=0.02,
                N_SAMPLES=args.n_samples_sde,
                EPSILON=1,
                TRAJ_LEN=args.traj_len,
                N_JOBS=-1,
                load_gen_f=f_model,
                mean_value=mean_value,
                std_value=std_value,
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
                SAVE_MODEL=args.save_model,
                device=device,
            )[0]
        learned_trans = LearnedTrans(model_g, mean_value, std_value, f_s=F_S)
    elif args.model == 'ens':
        print('\nTraining ensemble model')
        ens_params = dict(
            network_size=7, elite_size=5, state_size=F_S, action_size=F_A,
            hidden_size=200, use_decay=True, det_trans=True
        )
        ensblock = train_ensemble(
            env_class=env_class,
            env_params=env_params,
            ens_params=ens_params,
            project_name=project_name,
            name=args.model,
            N_SAMPLES=args.n_samples_ode * 20,
            BATCH_SIZE=args.model_batch_size,
            LR=args.model_lr,
            LOG_WANDB=LOG_WANDB,
            SAVE_MODEL=args.save_model,
            device=device,
        )

    #################
    # RL Training
    #################
    if args.train_rl:
        print("\nTraining RL Agent")

        # Create learned env
        if args.model == 'ode' or args.model == 'sde':
            learned_env = env_class(trans_model=learned_trans, **env_params)
            env = learned_env
            og_env = env_class(**env_params)
        else:
            og_env = env

        gamma = 0.99
        tau = 0.005
        policy = "Gaussian"
        target_update_interval = 1
        auto_entropy_tuning = False
        start_steps = 1
        updates_per_step = 1

        agent = SAC(
            num_inputs=env.observation_space.shape[0],
            action_space=env.action_space,
            gamma=gamma,
            tau=tau,
            policy=policy,
            target_update_interval=target_update_interval,
            automatic_entropy_tuning=auto_entropy_tuning,
            cuda=torch.cuda.is_available(),
            lr=args.lr,
            alpha=args.alpha,
            hidden_size=args.hidden_size,
        )

        if LOG_WANDB:
            wandb.init(
                project=f'rl-{repr(env)}',
                name=args.model,
                config={
                    "env_class": env_class.__name__,
                    "env_params": env_params,
                    "model_type": args.model,
                    'sac_agent_params': dict(
                        num_inputs=env.observation_space.shape[0],
                        action_space=env.action_space,
                        gamma=gamma,
                        tau=tau,
                        policy=policy,
                        target_update_interval=target_update_interval,
                        automatic_entropy_tuning=auto_entropy_tuning,
                        cuda=torch.cuda.is_available(),
                        lr=args.lr,
                        alpha=args.alpha,
                        hidden_size=args.hidden_size
                    ),
                    'train_agent_config': dict(
                        start_steps=start_steps,
                        batch_size=args.batch_size,
                        updates_per_step=updates_per_step,
                        num_steps=args.num_steps,
                        max_episode_len=args.max_episode_len,
                        replay_size=args.replay_size
                    )
                }
            )

        results = train_agent(
            agent,
            env,
            og_env=og_env,
            start_steps=start_steps,
            batch_size=args.batch_size,
            updates_per_step=updates_per_step,
            num_steps=args.num_steps,
            max_episode_len=args.max_episode_len,
            replay_size=args.replay_size,
            eval_deterministic=False,
            eval_freq=10,
            eval_n_episodes=25
        )

        # Final evaluation
        print('\nEvaluating on original env')
        final_eval_agent_len, final_eval_agent_rewards = eval_agent(
            agent, og_env, n_episodes=1000, determ=True
        )
        final_results_arr = {"final_eval_agent_len": final_eval_agent_len,
                             "final_eval_agent_rewards": final_eval_agent_rewards}
        final_results_reward_mean = np.mean(final_eval_agent_rewards)
        final_results_reward_std = np.std(final_eval_agent_rewards)
        results['final_results_arr'] = final_results_arr
        results['final_results_reward_mean'] = final_results_reward_mean
        results['final_results_reward_std'] = final_results_reward_std

        if LOG_WANDB:
            wandb.log({"final_results_reward_mean": final_results_reward_mean,
                       "final_results_reward_std": final_results_reward_std})

        if args.save_agent:
            print('Saving agent')
            dir = Path("checkpoints")
            dir.mkdir(exist_ok=True)
            run_id = wandb.run.id if LOG_WANDB and wandb.run else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = dir / f'{repr(env)}_{args.model}_{run_id}.pt'
            agent.save_checkpoint(env_name=repr(env), results=results, ckpt_path=ckpt_path)

        print(f"Final reward mean: {final_results_reward_mean:.4f} ± {final_results_reward_std:.4f}")

    if LOG_WANDB and wandb.run:
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    main()


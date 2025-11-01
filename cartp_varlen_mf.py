from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import wandb
from env.cartpole import CartPoleEnvStochVaryingLength, CartPoleEnvVaryingLength
from sac.sac import SAC
from utils.utils_train_sac import train_agent, eval_agent


# Env config
ENV_DICT = {'stoch': CartPoleEnvStochVaryingLength, 'det': CartPoleEnvVaryingLength}
ENV_PARAMS = {
    'stoch': dict(verbose=False, force_mag=10, diff_sigma=2.0),
    'det': dict(verbose=False, discrete=False)
}

parser = ArgumentParser(description="Train policy from scratch on target cartpole using same env data as model training.")
parser.add_argument('--env', choices=ENV_DICT.keys(), default='stoch',
                    help="Environment type: 'stoch' (default) or 'det'")
parser.add_argument('--length', type=float, default=3.0,
                    help="Target pole length (default: 3.0)")
parser.add_argument('--no-wandb', action='store_true', default=False,
                    help="Disable wandb logging")

parser.add_argument('--traj-len', type=int, default=20,
                    help="Trajectory length (default: 20)")
parser.add_argument('--n-samples', type=int, default=100,
                    help="Number of trajectories for target model training (default: 100)")

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


args = parser.parse_args()

wandb.login()
LOG_WANDB = not args.no_wandb
device = 'cuda' if torch.cuda.is_available() else "cpu"

print(f"Env: {args.env}")
print(f"Device: {device}")
print(f"Pole length: {args.length}\n")

env_class = ENV_DICT[args.env]
env_params = ENV_PARAMS[args.env]
env = env_class(**env_params)

# RL training
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
        project=f'rl-{repr(env)}-adaptation',
        name=f'scratch-trained-{args.length}',
        config = {
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
    start_steps=start_steps,
    batch_size=args.batch_size,
    updates_per_step=updates_per_step,
    num_steps=args.num_steps,
    max_episode_len=args.max_episode_len,
    replay_size=args.replay_size,
    eval_deterministic=False,
    cap_memory=True,
    max_memory_size=args.n_samples * args.traj_len,
)

# Final evaluation
eval_rewards = eval_agent(
    agent, env, n_episodes=1000, determ=True
)[1]
all_results = {"scratch_train_policy": eval_rewards,
               "length": args.length}
eval_rewards_mean = np.mean(eval_rewards)
eval_rewards_std = np.std(eval_rewards)
print(f"Scratch trained policy: {eval_rewards_mean:.2f} ± {eval_rewards_std:.2f}")

if LOG_WANDB:
    wandb.log({"eval_rewards_mean": eval_rewards_mean,
               "eval_rewards_std": eval_rewards_std})

# Save results
dir = Path(f'results_adaptation/{repr(env)}')
dir.mkdir(exist_ok=True, parents=True)
file_name = f'results_scratch_train_{args.length}_{wandb.run.id}.pt'
torch.save(all_results, dir / file_name)
print(f"\nSaved results to {dir / file_name}")
print('Done!')

wandb.finish()
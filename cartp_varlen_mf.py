from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import wandb
from env.cartpole import CartPoleEnvStochVaryingLength
from sac.sac import SAC
from utils.utils_train_sac import train_agent, eval_agent


wandb.login()
device = 'cuda' if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--length', default=1.2, type=float)
args = parser.parse_args()
print(args.length)
pole_length = args.length

# env_class = CartPoleEnvVaryingLength
# env_params = dict(pole_len=pole_length)
env_class = CartPoleEnvStochVaryingLength
env_params = dict(force_mag=10, diff_sigma=2.0, pole_len=pole_length)

replay_size = 300000
start_steps = 1
batch_size = 32
updates_per_step = 1
gamma = 0.99
tau = 0.005
policy = "Gaussian"
target_update_interval = 1
auto_entropy_tuning = False
cuda=True
lr=1e-3
alpha=0.2
hidden_size = 200
num_steps = 500
max_episode_len = 500
env = env_class(**env_params)
LOG_WANDB = True

agent = SAC(
    num_inputs=env.observation_space.shape[0],
    action_space=env.action_space,
    gamma=gamma,
    tau=tau,
    policy=policy,
    target_update_interval=target_update_interval,
    automatic_entropy_tuning=auto_entropy_tuning,
    cuda=cuda,
    lr=lr,
    alpha=alpha,
    hidden_size=hidden_size,
)

if LOG_WANDB:
    wandb.init(
        project=f'rl-{repr(env)}-mf',
        name=f'mf-{pole_length}',
        config = {
            'sac_agent_params': dict(
                num_inputs=env.observation_space.shape[0],
                action_space=env.action_space,
                gamma=gamma,
                tau=tau,
                policy=policy,
                target_update_interval=target_update_interval,
                automatic_entropy_tuning=auto_entropy_tuning,
                cuda=cuda,
                lr=lr,
                alpha=alpha,
                hidden_size=hidden_size
            ),
            'train_agent_config': dict(
                start_steps=start_steps,
                batch_size=batch_size,
                updates_per_step=updates_per_step,
                num_steps=num_steps,
                max_episode_len=max_episode_len,
                replay_size=replay_size
            )
        }
    )

results = train_agent(
    agent,
    env,
    start_steps=start_steps,
    batch_size=batch_size,
    updates_per_step=updates_per_step,
    num_steps=num_steps,
    max_episode_len=max_episode_len,
    replay_size=replay_size,
    eval_deterministic=False,
    cap_memory=True,
    max_memory_size=100 * 20,
)

# Eval Agent Performance after training
# (n_episodes,)
final_eval_agent_len, final_eval_agent_rewards = eval_agent(
    agent, env, n_episodes=1000, determ=True
)
all_results = {"retrain_policy": final_eval_agent_rewards,
               "length": pole_length}
final_results_reward_mean = np.mean(final_eval_agent_rewards)
final_results_reward_std = np.std(final_eval_agent_rewards)
print(final_results_reward_mean, final_results_reward_std)

if LOG_WANDB:
    wandb.log({"final_results_reward_mean": final_results_reward_mean,
               "final_results_reward_std": final_results_reward_std})

dir = Path(f'results_adaptation/{repr(env)}')
dir.mkdir(exist_ok=True, parents=True)
torch.save(all_results, dir/f'results_retrain_mf_{pole_length}_{wandb.run.id}.pt')

print('Saving agent')
dir = Path("checkpoints")
dir.mkdir(exist_ok=True)
ckpt_path = dir / f'{repr(env)}_{pole_length}_mf_{wandb.run.id}.pt'
agent.save_checkpoint(env_name=repr(env), results=results, ckpt_path=ckpt_path)

wandb.finish()
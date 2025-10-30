import itertools
import numpy as np
import torch

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
import wandb
import math


@torch.no_grad()
def eval_agent(agent, og_env, n_episodes, determ=True):
    episode_len_lst = []
    reward_lst = []
    for _ in range(n_episodes):
        state = og_env.reset()[0]
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0

        while not done and not truncated:
            action = agent.select_action(state, evaluate=determ)
            next_state, reward, done, truncated, _ = og_env.step(action)
            episode_reward += reward
            episode_steps += 1
            state = next_state

        episode_len_lst.append(episode_steps)
        reward_lst.append(episode_reward)
    episode_len_lst, reward_lst = np.array(episode_len_lst), np.array(reward_lst)
    return episode_len_lst, reward_lst


@torch.no_grad()
def eval_agent_invdyn(
        agent_src, env_tar, trans_src, inv_model, device, n_episodes, determ=True,
        use_delta_a=False, rms=False, rms_s_tar=None, rms_s_next_src=None,
):
    episode_len_lst = []
    reward_lst = []

    for _ in range(n_episodes):
        s_tar = env_tar.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0

        while not terminated and not truncated:
            a_src = agent_src.select_action(s_tar, evaluate=determ)
            s_next_src = trans_src(s_tar, a_src)

            a_src_ts = torch.from_numpy(a_src).float().to(device)  # (F_A,)
            s_tar_ts = torch.from_numpy(s_tar).float().to(device)  # (F_S,)
            s_next_src_ts = torch.from_numpy(s_next_src).float().to(device)  # (F_S,)

            if rms:
                assert rms_s_tar is not None and rms_s_next_src is not None, \
                    "You must pass rms_s_tar/rms_s_next_src when rms=True"
                # Normalize inputs (1, F_S)
                s_tar_in = rms_s_tar.normalize(s_tar_ts).unsqueeze(0)
                s_next_src_in = rms_s_next_src.normalize(s_next_src_ts).unsqueeze(0)
            else:
                # Use raw inputs (1, F_S)
                s_tar_in = s_tar_ts
                s_next_src_in = s_next_src_ts

            if use_delta_a:
                delta_a_ts = inv_model(s_tar_in, s_next_src_in)
                a_tar_ts = a_src_ts + delta_a_ts  # (1, F_A)
            else:
                a_tar_ts = inv_model(s_tar_in, s_next_src_in)  # (1, F_A)
            a_tar = a_tar_ts.cpu().numpy().squeeze(0)

            s_next_tar, reward, terminated, truncated, _ = env_tar.step(a_tar)
            episode_reward += reward
            episode_steps += 1
            s_tar = s_next_tar

        episode_len_lst.append(episode_steps)
        reward_lst.append(episode_reward)

    episode_len_lst, reward_lst = np.array(episode_len_lst), np.array(reward_lst)
    return episode_len_lst, reward_lst


def train_agent(
        agent: SAC,
        env,
        start_steps,
        batch_size,
        updates_per_step,
        num_steps,
        max_episode_len,
        replay_size,
        og_env=None,
        eval_n_episodes=10,
        eval_freq=10,
        eval_deterministic=False,
        cap_memory=False,
        max_memory_size=None,
):
    memory = ReplayMemory(replay_size, seed=0)

    total_numsteps = 0
    updates = 0
    reward_lst = []
    episode_len_lst = []

    eval_mean_len_lst = []
    eval_mean_reward_lst = []
    eval_time_lst = []

    det_eval_mean_len_lst = []
    det_eval_mean_reward_lst = []
    det_eval_time_lst = []

    for i_episode in itertools.count(1):
        if total_numsteps > num_steps:
            break

        episode_reward = 0
        episode_steps = 0
        done, truncated = False, False
        state, _ = env.reset()

        while not done and not truncated:
            if start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:
                # Number of updates per step in environment
                for i in range(updates_per_step):
                    # Update parameters of the agent
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = (
                        agent.update_parameters(memory, batch_size, updates)
                    )
                    updates += 1

            next_state, reward, done, truncated, _ = env.step(action)  # Step
            episode_steps += 1
            episode_reward += reward

            mask = float(not done)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            if cap_memory and len(memory) > max_memory_size:
                # force exit inner loops
                print(f"Reached max memory ({len(memory)} > {max_memory_size}), stopping collection.")
                break

            state = next_state
            if episode_steps >= max_episode_len:
                break

        if cap_memory and len(memory) > max_memory_size:
            # force exit outer loops
            print("Exiting training as cap_memory hit.")
            break

        total_numsteps += 1
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        # Evaluate model on original environment
        if og_env is not None and i_episode % eval_freq == 0:
            eval_mean_len, eval_mean_reward = eval_agent(agent, og_env, eval_n_episodes, determ=True)
            eval_mean_len, eval_mean_reward = eval_mean_len.mean(), eval_mean_reward.mean()
            eval_mean_len_lst.append(eval_mean_len)
            eval_mean_reward_lst.append(eval_mean_reward)
            eval_time_lst.append(i_episode)
            wandb.log({'eval_reward': eval_mean_reward,
                       'eval_len': eval_mean_len,
                       'episode_steps': episode_steps})
            print(f"Evaluation: episode steps: {eval_mean_len}, reward: {round(eval_mean_reward, 2)}")

        # Agent's Deterministic Performance
        if eval_deterministic:
            det_mean_len, det_mean_reward = eval_agent(agent, env, n_episodes=100, determ=True)
            det_mean_len, det_mean_reward = det_mean_len.mean(), det_mean_reward.mean()
            det_eval_mean_len_lst.append(det_mean_len)
            det_eval_mean_reward_lst.append(det_mean_reward)
            det_eval_time_lst.append(i_episode)
            if wandb.run is not None:
                wandb.log({"num_steps": total_numsteps,
                           "det_eval_reward": det_mean_reward,
                           "det_mean_len": det_mean_len})
            print(f"Evaluation (Deterministic): episode steps: {det_mean_len}, reward: {round(det_mean_reward, 2)}")

        if wandb.run is not None:
            wandb.log({"num_steps": total_numsteps,
                       "episode_reward": episode_reward,
                       "episode_steps": episode_steps})
        reward_lst.append(episode_reward)
        episode_len_lst.append(episode_steps)

    return dict(episode_len_lst=episode_len_lst,
                reward_lst=reward_lst,
                eval_mean_len_lst=eval_mean_len_lst,
                eval_mean_reward_lst=eval_mean_reward_lst,
                eval_time_lst=eval_time_lst,
                det_eval_mean_len_lst=det_eval_mean_len_lst,
                det_eval_mean_reward_lst = det_eval_mean_reward_lst,
                det_eval_time_lst = det_eval_time_lst)

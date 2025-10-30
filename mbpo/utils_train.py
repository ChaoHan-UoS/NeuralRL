from itertools import count

import numpy as np
import wandb

from sac.replay_memory import ReplayMemory


def train_mbpo(args, env_sampler, predict_env, agent, env_pool, model_pool):
    total_step = 0  # global counter of real-env steps
    reward_sum = 0
    rollout_length = 1
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step  # current real-env steps in the epoch
            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                # Train the model on all data in env buffer
                train_predict_model(args, env_pool, predict_env)

                # Schedule rollout length
                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                # Generate model rollouts
                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)

            # Take one env step under the current policy and append it to env buffer
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)

            # Policy update
            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(
                    args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent
                )
            total_step += 1

            # End-of-epoch evaluation
            if total_step % args.epoch_length == 0:
                eval_reward = eval_agent(env_sampler, agent, n_eval_episodes=25)
                if wandb.run is not None:
                    wandb.log(
                        dict(
                            total_env_steps=total_step,
                            eval_rewards=eval_reward,
                            epoch=epoch_step,
                        )
                    )
                print(f"Epoch: {epoch_step + 1}, Step: {total_step}, Eval Reward: {eval_reward}")


def eval_agent(env_sampler, agent, n_eval_episodes=25):
    """
    Evaluate the agent on the environment for `n_eval_episodes` episodes.
    Returns the average reward over all episodes.
    """
    total_reward = 0.0
    for _ in range(n_eval_episodes):
        env_sampler.current_state = None
        done = False
        episode_reward = 0.0

        while not done:
            cur_state, action, next_state, reward, done, info = \
                env_sampler.sample(agent, eval_t=True)
            episode_reward += reward
        total_reward += episode_reward

    avg_reward = total_reward / n_eval_episodes
    return avg_reward


def exploration_before_start(args, env_sampler, env_pool, agent):
    """
    Collect `init_exploration_steps` transitions with the (initially random) policy,
    filling env_pool so the model has data to train on.
    """
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    """
    Compute a new horizon (linearly increasing between `rollout_min_length` and rollout_max_length` over epochs).
    """
    rollout_length = min(
        max(
            args.rollout_min_length + (epoch_step - args.rollout_min_epoch) /
            (args.rollout_max_epoch - args.rollout_min_epoch) *
            (args.rollout_max_length - args.rollout_min_length),
            args.rollout_min_length
        ),
        args.rollout_max_length
    )
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    """Fit ensemble dynamics model on all data in `env_pool`."""
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)  # (N, s_dim + a_dim)
    labels = np.concatenate(
        (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1
    )  # (N, r_dim + s_dim)

    predict_env.model.train(inputs, labels, batch_size=args.model_batch_size, holdout_ratio=0.2)


def resize_model_pool(args, rollout_length, model_pool):
    """
    Resize the model buffer to hold `model_retain_epochs` epochs * `model_steps_per_epoch` steps
    worth of model transitions.
    """
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)
    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    # Sample a batch of transitions, only use the initial states
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        action = agent.select_action(state)  # (bs, a_dim)
        # Predict and push a batch of transitions
        next_states, rewards, terminals, info = predict_env.step(state, action)
        model_pool.push_batch(
            [(state[j], action[j], rewards[j], next_states[j], terminals[j])
             for j in range(state.shape[0])]
        )
        nonterm_mask = ~terminals.squeeze(-1)
        # Break when ALL episodes in the batch terminate
        if nonterm_mask.sum() == 0:
            break
        # bs shrinks as episodes terminate
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0
    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        # Batch data mixed from real and model ones
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        # Sample the real transitions
        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            # Sample the model transitions
            model_state, model_action, model_reward, model_next_state, model_done = \
                model_pool.sample_all_batch(int(model_batch_size))
            # Concatenate with the real transitions
            batch_state = np.concatenate((env_state, model_state), axis=0)
            batch_action = np.concatenate((env_action, model_action), axis=0)
            batch_reward = np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0
            )
            batch_next_state = np.concatenate((env_next_state, model_next_state), axis=0)
            batch_done = np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0
            )
        else:
            # Fallback: only real transition this policy iteration
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)  # non-terminal mask
        agent.update_parameters(
            (batch_state, batch_action, batch_reward, batch_next_state, batch_done),
            args.policy_train_batch_size, i, is_mem_batch=True
        )

    return args.num_train_repeat
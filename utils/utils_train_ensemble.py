import datetime
import numpy as np
import wandb
# import fire

from utils.utils import save_model_ens
from sac.replay_memory import ReplayMemory as ReplayMemory_mbpo
from mbpo.model import EnsembleBlock
from mbpo.sample_env import EnvSampler


def train_ensemble(
        env_class,
        env_params,
        ens_params,
        project_name="model-train-stoch-cartpole-ensemble",
        name="ensemble",
        N_SAMPLES=100000,
        load_block=None,
        BATCH_SIZE=128,
        LR=8e-4,
        LOG_WANDB=True,
        SAVE_MODEL=True,
        device="cuda",
):
    env = env_class(**env_params)
    env_sampler = EnvSampler(env)
    env_pool = ReplayMemory_mbpo(capacity=1000000)
    block = EnsembleBlock(**ens_params) if load_block is None else load_block

    if LOG_WANDB:
        wandb.init(
            project=project_name,
            name=name,
            config={
                "env_class": env_class,
                "env_params": env_params,
                "ens_params": ens_params,
                "n_samples": N_SAMPLES,
                "lr": LR,
                "batch_size": BATCH_SIZE,
            },
        )

    # Randomly collect data for training
    for i in range(N_SAMPLES):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent=None)
        env_pool.push(cur_state, action, reward, next_state, done)

    # Train the model
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)  # (N, s_dim + a_dim)
    labels = np.concatenate(
        (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1
    )  # (N, r_dim + s_dim)
    block.train(inputs, labels, batch_size=BATCH_SIZE, holdout_ratio=0.2)

    # Save model
    if SAVE_MODEL:
        run_id = wandb.run.id if wandb.run is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        to_add = dict()
        to_add["ens_params"] = ens_params
        save_model_ens(block, repr(env), run_id=run_id, **to_add)

    wandb.finish()
    return block


# if __name__ == "__main__":
#     wandb.login()
#     fire.Fire(train_ensemble)
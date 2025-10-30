from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from joblib import Parallel, delayed
import h5py
from utils.utils import standardise, unstandardise


def get_min_max_vel(train_dataset):
    # Dataset: B, T, F_s
    min_lim_vel = train_dataset[..., 1].min()
    max_lim_vel = train_dataset[..., 1].max()
    return min_lim_vel, max_lim_vel


def run_episode_arr(env, epsilon=0.5, boring_policy=False):
    """
    epsilon=1: completely random
    epsilon=0: completely deterministic

    Return state and action sequences
    Both sequences have the same length. A zero action is added at the end of
    the action array to make sure of that.
    """
    done = False
    truncated = False

    obs = env.reset()[0]
    params = np.ones_like(obs)
    # state_action = []
    state = []
    actions = []

    while not (done or truncated):  # loop until episode is done
        # Choose a random agent
        # to_add: [s_t, a_t, s_t+1, t]
        if boring_policy:
            action = np.array(0)
        else:
            if np.random.random() <= epsilon:
                # a random action
                action = env.action_space.sample()
            else:
                # a binary action from a deterministic linear policy (for cartpole)
                action = 1 if np.dot(obs, params) > 0 else 0
            action = np.array(action)
        state.append(obs)
        actions.append(action)

        obs, reward, done, truncated, _ = env.step(action)
    # State: T*F_s
    # Actions: T*F_a
    state.append(obs)

    # Convert everything to numpy
    state = np.array(state)
    actions = np.array(actions)
    if actions.ndim == 1:
        # Add extra dimension to actions if one dimensional
        actions = actions[..., None]
    actions = np.concatenate([actions, np.zeros((1, actions.shape[-1]))])
    return state, actions


def generate_dataset(env, n_samples=100, epsilon=0.5, traj_min_len=50):
    # from integ_model_learning_2_actor_critic.ipynb
    states_lst = []
    actions_lst = []
    for _ in tqdm(range(n_samples)):
        states, actions = run_episode_arr(env, epsilon=epsilon)  # (TRAJ_LEN, F_s), (TRAJ_LEN, F_a)
        while states.shape[0] < traj_min_len:
            states, actions = run_episode_arr(env, epsilon=epsilon)
        if states.shape[0] > traj_min_len:
            s_idx = np.random.choice(states.shape[0] - traj_min_len)
            states = states[s_idx : s_idx + traj_min_len]
            actions = actions[s_idx : s_idx + traj_min_len]
            # print(states.shape[0])
        states_lst.append(states)
        actions_lst.append(actions)
    # dataset: N, TRAJ_MIN_LEN, F_s + F_a
    dataset = np.concatenate((states_lst, actions_lst), axis=-1)
    dataset = torch.from_numpy(dataset).to(torch.float)
    return dataset

def generate_dataset_oversample(env, n_samples=100, epsilon=0.5, traj_min_len=50, oversample=1):
    # from integ_model_learning_2_actor_critic.ipynb
    # Generate multiple sub-trajectories from a single trajectory; used when trajectories are long
    states_lst = []
    actions_lst = []
    # for _ in tqdm(range(n_samples)):
    # n_samples = n_samples
    while n_samples > 0:
        states, actions = run_episode_arr(env, epsilon=epsilon)
        while states.shape[0] < traj_min_len:
            states, actions = run_episode_arr(env, epsilon=epsilon)
        if states.shape[0] > traj_min_len:
            s_idxs = np.random.choice(states.shape[0] - traj_min_len, size=min(oversample, n_samples))
            # print(s_idxs)
            for s_idx in s_idxs:
                # print(s_idx+traj_min_len)
                state_samp = states[s_idx : s_idx + traj_min_len]
                assert state_samp.shape[0] == traj_min_len, state_samp.shape[0]
                action_samp = actions[s_idx : s_idx + traj_min_len]
                assert action_samp.shape[0] == traj_min_len, action_samp.shape[0]
                states_lst.append(state_samp)
                actions_lst.append(action_samp)
                n_samples -= 1
        else:
            states_lst.append(states)
            actions_lst.append(actions)
    # dataset: N, TRAJ_MIN_LEN, F_s + F_a
    dataset = np.concatenate((states_lst, actions_lst), axis=-1)
    dataset = torch.from_numpy(dataset).to(torch.float)
    return dataset


def generate_dataset_parallel(
    env_class, env_params, n_samples=100, epsilon=0.5, traj_min_len=50, n_jobs=1
) -> torch.Tensor:

    def single_job():
        # Create a copy of env to prevent interleaving between jobs
        # env = env_class(env_params)
        env = create_env(env_class, env_params, max_episode_steps=1000)
        states, actions = run_episode_arr(env, epsilon=epsilon)
        while states.shape[0] < traj_min_len:
            states, actions = run_episode_arr(env, epsilon=epsilon)
        if states.shape[0] > traj_min_len:
            s_idx = np.random.choice(states.shape[0] - traj_min_len)
            states = states[s_idx : s_idx + traj_min_len]
            actions = actions[s_idx : s_idx + traj_min_len]
            # print(states.shape[0])
            # states_lst.append(states)
            # actions_lst.append(actions)
        return states, actions

    # n_samples tasks (trajectories) are distributed dynamically across n_jobs workers
    # out: [(states, actions), ...]
    out = Parallel(n_jobs=n_jobs)(delayed(single_job)() for _ in range(n_samples))
    states_lst = list(map(lambda x: x[0], out))
    actions_lst = list(map(lambda x: x[1], out))
    # dataset: N, TRAJ_MIN_LEN, F_s + F_a
    dataset = np.concatenate((states_lst, actions_lst), axis=-1)
    dataset = torch.from_numpy(dataset).to(torch.float)
    return dataset


class TrajDataset(Dataset):
    # from integ_model_learning_2_actor_critic.ipynb
    def __init__(self, data, f_s=4):
        self.data = data  # data: N, T, F_s + F_a
        self.f_s = f_s
        self.data = data.transpose(-1, -2)  # data: N, F_s + F_a, T

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        samp = self.data[index]
        # s_0: (F_s, ), actions: (F_a, T), target: (F_s, T)
        return samp[:self.f_s, 0], samp[self.f_s:], samp[:self.f_s]


def get_inf_iterloader(data_loader):
    while True:
        for batch in data_loader:
            yield batch
            
            
def get_mean_std(dataset, f_s, standardise_in_signal=True):
    """
    Args:
        dataset (_type_):
        f_s (int, optional): _description_. Defaults to 4.
    Returns:
        _type_: _description_
    """
    # dataset: B, T, F_s + F_a
    if standardise_in_signal:
        mean_value = dataset.mean(dim=(0, 1))
        std_value = dataset.std(dim=(0, 1))
        return mean_value, std_value
    # Only standardise observational space
    f_a = dataset.shape[-1] - f_s
    mean_value = dataset[..., :f_s].mean(dim=(0, 1))
    std_value = dataset[..., :f_s].std(dim=(0, 1))
    mean_value = torch.concatenate([mean_value, torch.zeros(f_a)])
    std_value = torch.concatenate([std_value, torch.ones(f_a)])
    return mean_value, std_value


def get_dataset_from_h5py(trainset_path):
    assert Path(trainset_path).exists(), f"Path: {trainset_path} does not exist"
    with h5py.File(trainset_path, "r") as f:
        train_dataset = torch.from_numpy(np.copy(f["dataset"]))
    return train_dataset


from gymnasium.wrappers import TimeLimit, OrderEnforcing

def create_env(env_class, env_params, max_episode_steps=1000):
    # The TimeLimit is required to limit the number of steps of the environment
    return TimeLimit(
        OrderEnforcing(env_class(**env_params)),
        max_episode_steps=max_episode_steps,
    )


def unstandardise_state(state, mean_value, std_value):
    assert state.ndim == 1
    f_s = state.shape[0]
    return unstandardise(state, mean_value[:f_s], std_value[:f_s])


def standardise_state(state, mean_value, std_value):
    assert state.ndim == 1
    f_s = state.shape[0]
    return standardise(state, mean_value[:f_s], std_value[:f_s])
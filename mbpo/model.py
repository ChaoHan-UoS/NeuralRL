import math
import gzip
import itertools

import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import wandb


device = torch.device('cuda')

num_train = 60000  # 60k train examples
num_test = 10000  # 10k test examples
train_inputs_file_path = './MNIST_data/train-images-idx3-ubyte.gz'
train_labels_file_path = './MNIST_data/train-labels-idx1-ubyte.gz'
test_inputs_file_path = './MNIST_data/t10k-images-idx3-ubyte.gz'
test_labels_file_path = './MNIST_data/t10k-labels-idx1-ubyte.gz'

BATCH_SIZE = 100


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
            self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        # Weight decay for L2 regularization of weights to avoid overfitting
        # by penalizing large weight values during training
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features, device=device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ensemble_size fully connected layers simultaneously.
        Args:
            input: (ensemble_size, batch_size, in_features)
        Returns: (ensemble_size, batch_size, out_features)
        """
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(
            self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False
    ):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance (diagonal covariance matrix) output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        # Fixed the upper and lower bounds for logvar
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()  # Swish activation func

    def forward(self, x, ret_log_var=False):
        # x: tensor(ensemble_size, batch_size, state_size + action_size)
        # 5-layer MLP for each ensemble
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]
        # Differentially implement min(x, max_logvar) to ensure logvar <= max_logvar
        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        # Differentially implement max(x, min_logvar) to ensure logvar >= min_logvar
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        # mean, logvar, labels: (ensemble_size, N, dim)
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)  # 1 / σ^2
        if inc_var_loss:
            mse_loss = torch.mean(
                torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1),
                dim=-1
            )  # (ensemble_size,)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)  # (ensemble_size,)
            # Heteroscedastic diagonal Gaussian NLL: NLL(y | μ, σ^2) ∝ [(y - μ)^2 / σ^2 + log(σ^2)]
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            # Homoscedastic diagonal Gaussian NLL (constant variance): MSE = (y - μ)^2
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))  # (ensemble_size,)
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        # Encourage lower upper bound and higher lower bound for predicted logvar
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel():
    def __init__(
            self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False
    ):
        self.network_size = network_size
        self.elite_size = elite_size
        # self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(
            state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay
        )
        self.scaler = StandardScaler()  # inputs normalization

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        """
        Train ensemble nets, each net predicts concat(s, a) -> a Gaussian over concat(rew, next_s - s)
        Args:
            inputs: array(N, s_dim + a_dim)
            labels: array(N, r_dim + s_dim)
            holdout_ratio: fraction of eval data for elite selection and early stopping
            max_epochs_since_update: max number of epochs without improvement before stopping training
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        # self._state = {}
        # (best_epoch, best_loss) for each ensemble net i
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        # Holdout split over dataset size N
        num_holdout = int(inputs.shape[0] * holdout_ratio)  # H
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]  # (N - H, s_dim + a_dim / r_dim + s_dim)
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]  # (H, s_dim + a_dim / r_dim + s_dim)

        # Normalize inputs but raw labels
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        # Broadcast holdout to the ensemble size B
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])  # tensor(B, H, s_dim + a_dim)
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])  # tensor(B, H, r_dim + s_dim)

        for epoch in itertools.count():
            train_idx = np.vstack(
                [np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)]
            )                                                                         # (B, N - H)
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: (start_pos + batch_size)]               # (B, bs)
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)  # tensor(B, bs, s_dim + a_dim)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)  # tensor(B, bs, r_dim + s_dim)

                # Forward all ensemble nets at once and backprop
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)     # (B, bs, r_dim + s_dim)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)         # scalar
                self.ensemble_model.train(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)  # (B, H, r_dim + s_dim)

                # Rank per-ensemble point-prediction accuracy for elite selection
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False
                )                                                                     # (B, )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

                # Early stopping
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

            holdout_mse_mean = np.mean(holdout_mse_losses).item()
            if wandb.run is not None:
                wandb.log(dict(holdout_mse_mean=holdout_mse_mean))
            print('Model Epoch: {}, Holdout MSE Mean: {}'.format(epoch, holdout_mse_mean))
        print('\n')

    def _save_best(self, epoch, holdout_losses):
        """ Early stopper
        Training continues as long as at least one ensemble net improves on the holdout MSE.
        """
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        """
        Predict an ensemble B of N Gaussian's over concat(rew, next_s - s), given N input concat(s, a)
        array(N, s_dim + a_dim) -> array(B, N, r_dim + s_dim)
        """
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(
                inputs[i: min(i + batch_size, inputs.shape[0])]
            ).float().to(device)                   # (bs, s_dim + a_dim)
            b_mean, b_var = self.ensemble_model(
                input[None, :, :].repeat([self.network_size, 1, 1]),
                ret_log_var=False
            )                                      # (B, bs, r_dim + s_dim)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)   # (B, N, r_dim + s_dim)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var


class EnsembleBlock(EnsembleDynamicsModel):
    def __init__(
            self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False,
            det_trans=False, differ_trans=False
    ):
        super(EnsembleBlock, self).__init__(
            network_size, elite_size, state_size, action_size, reward_size, hidden_size, use_decay
        )
        self.det_trans = det_trans
        self.differ_trans = differ_trans

    def __call__(self, obs, act):
        if self.differ_trans:
            return self.forward_differ(obs, act)
        else:
            return self.forward(obs, act)

    def forward(self, obs, act):
        if len(obs.shape) == 1:
            # add bs = 1
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)                # (bs, s_dim + a_dim)
        ensemble_model_means, ensemble_model_vars = self.predict(inputs)  # (B, bs, r_dim + s_dim)
        ensemble_model_means[:, :, 1:] += obs                             # s' = s + delta_s
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if self.det_trans:
            ensemble_samples = ensemble_model_means                       # (B, bs, r_dim + s_dim)
        else:
            ensemble_samples = ensemble_model_means + \
                               np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        # Randomly choose one elite model for each sample in the batch
        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.elite_model_idxes, size=batch_size)  # (bs, )
        batch_idxes = np.arange(0, batch_size)                                   # (bs, )
        samples = ensemble_samples[model_idxes, batch_idxes]                     # (bs, r_dim + s_dim)
        next_obs = samples[:, 1:]                                                # (bs, s_dim)

        if return_single:
            next_obs = next_obs[0]

        return next_obs

    def forward_differ(self, obs, act):
        """Differentiable version of forward using PyTorch operations"""
        # Handle both single observation and batch
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float32, device=device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
            act = act.unsqueeze(0)
            return_single = True
        else:
            return_single = False

        # Concatenate observation and action
        inputs = torch.cat([obs, act], dim=-1)  # (bs, s_dim + a_dim)

        # Normalize inputs
        mu = torch.tensor(self.scaler.mu, dtype=torch.float32, device=device)
        std = torch.tensor(self.scaler.std, dtype=torch.float32, device=device)
        inputs_normalized = (inputs - mu) / std

        # Forward pass through ensemble model
        inputs_normalized = inputs_normalized.unsqueeze(0).repeat(self.network_size, 1, 1)
        mean, var = self.ensemble_model(inputs_normalized, ret_log_var=False)

        # Add delta_s to current state: s' = s + delta_s
        mean[:, :, 1:] += obs.unsqueeze(0).repeat(self.network_size, 1, 1)

        # Get samples
        if self.det_trans:
            samples = mean  # Deterministic transition
        else:
            # Generate random noise for stochastic transition
            noise = torch.randn_like(mean) * torch.sqrt(var)
            samples = mean + noise

        # For backpropagation, we use mean of elite models (differentiable)
        elite_models = torch.tensor(self.elite_model_idxes, dtype=torch.long, device=device)
        next_obs_and_reward = torch.mean(samples[elite_models], dim=0)

        # Extract next observation (skip reward)
        next_obs = next_obs_and_reward[:, 1:]

        if return_single:
            next_obs = next_obs[0]

        return next_obs


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


def get_data(inputs_file_path, labels_file_path, num_examples):
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8) / 255.0
        inputs = data.reshape(num_examples, 784)

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_examples)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int8)


def set_tf_weights(model, tf_weights):
    print(tf_weights.keys())
    pth_weights = {}
    pth_weights['max_logvar'] = tf_weights['BNN/max_log_var:0']
    pth_weights['min_logvar'] = tf_weights['BNN/min_log_var:0']
    pth_weights['nn1.weight'] = tf_weights['BNN/Layer0/FC_weights:0']
    pth_weights['nn1.bias'] = tf_weights['BNN/Layer0/FC_biases:0']
    pth_weights['nn2.weight'] = tf_weights['BNN/Layer1/FC_weights:0']
    pth_weights['nn2.bias'] = tf_weights['BNN/Layer1/FC_biases:0']
    pth_weights['nn3.weight'] = tf_weights['BNN/Layer2/FC_weights:0']
    pth_weights['nn3.bias'] = tf_weights['BNN/Layer2/FC_biases:0']
    pth_weights['nn4.weight'] = tf_weights['BNN/Layer3/FC_weights:0']
    pth_weights['nn4.bias'] = tf_weights['BNN/Layer3/FC_biases:0']
    pth_weights['nn5.weight'] = tf_weights['BNN/Layer4/FC_weights:0']
    pth_weights['nn5.bias'] = tf_weights['BNN/Layer4/FC_biases:0']
    for name, param in model.ensemble_model.named_parameters():
        if param.requires_grad:
            # print(name)
            print(param.data.shape, pth_weights[name].shape)
            param.data = torch.FloatTensor(pth_weights[name]).to(device).reshape(param.data.shape)
            pth_weights[name] = param.data
            print(name)


def main():
    torch.set_printoptions(precision=7)
    import pickle
    # Import MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    # train_inputs, train_labels = get_data(train_inputs_file_path, train_labels_file_path, num_train)
    # test_inputs, test_labels = get_data(test_inputs_file_path, test_labels_file_path, num_test)

    num_networks = 7
    num_elites = 5
    state_size = 17
    action_size = 6
    reward_size = 1
    pred_hidden_size = 200
    model = EnsembleDynamicsModel(num_networks, num_elites, state_size, action_size, reward_size, pred_hidden_size)

    # load tf weights and set it to be the inital weights for pytorch model
    with open('tf_weights.pkl', 'rb') as f:
        tf_weights = pickle.load(f)
    # set_tf_weights(model, tf_weights)
    # x = model.model_list[0].named_parameters()
    # for name, param in model.model_list[0].named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # exit()
    BATCH_SIZE = 5250
    import time
    st_time = time.time()
    with open('test.npy', 'rb') as f:
        train_inputs = np.load(f)
        train_labels = np.load(f)
    for i in range(0, 1000, BATCH_SIZE):
        # train_inputs = np.random.random([BATCH_SIZE, state_size + action_size])
        # train_labels = np.random.random([BATCH_SIZE, state_size + 1])
        model.train(train_inputs, train_labels, holdout_ratio=0.2)
        # mean, var = model.predict(train_inputs[:100])
        # print(mean[0])
        # print(mean.mean().item())
        # print(var[0])
        # print(var.mean().item())
        # exit()
    print(time.time() - st_time)
    # for name, param in model.model_list[0].named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape,param)
    exit()
    # for i in range(0, 10000, BATCH_SIZE):
    #     model.train(Variable(torch.from_numpy(train_inputs[i:i + BATCH_SIZE])), Variable(torch.from_numpy(train_labels[i:i + BATCH_SIZE])))
    #
    # model.predict(Variable(torch.from_numpy(test_inputs[:1000])))


if __name__ == '__main__':
    main()

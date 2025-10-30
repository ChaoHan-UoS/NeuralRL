import torch
from torch import nn
from utils.utils import MLP, standardise, unstandardise
from utils.cu_odeint import odeint_w_in_signal, sdeint_noise_w_in_signal, get_solver


class InverseDynamicsModel(MLP):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, num_layers=2, use_delta_a=False):
        super().__init__(state_dim*2, action_dim, hidden_sizes, num_layers, tanh=True)
        self.use_delta_a = use_delta_a

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=-1)
        y = super().forward(x)
        return 2 * y if self.use_delta_a else y


class ODEFunc(nn.Module):
    # Derivative of the ODE
    def __init__(self, f_model: MLP):
        super().__init__()
        self.f_model = f_model

    def forward(self, x0, in_signal, t):
        """
        Args:
            x0 (torch.Tensor): (B, F_s)
            in_signal (torch.Tensor): (B, F_a)
            t (torch.Tensor): (1, )
        return: (B, F_s)
        """
        s = torch.cat([in_signal, x0], dim=-1)
        return self.f_model(s)


class AugFModel(nn.Module):
    def __init__(self, base_f_model: MLP, remap_func: MLP, in_signal_dim: int):
        super().__init__()
        self.base = base_f_model
        self.remapper = remap_func
        self.in_signal_dim = in_signal_dim

    def forward(self, s: torch.Tensor):
        # s has shape (B, F_a + F_s)
        f_out = self.base(s)                            # (B, F_s)
        in_signal = s[:, :self.in_signal_dim]           # (B, F_a)
        return self.remapper(torch.cat([f_out, in_signal], dim=-1))


class AugODEFunc(ODEFunc):
    # Augmented MLP as the derivative of an ODE for target transition dynamics
    def __init__(self, base_f_model: MLP, remap_func: MLP, in_signal_dim: int):
        aug_f_model = AugFModel(base_f_model, remap_func, in_signal_dim)
        super().__init__(aug_f_model)
        self.base_f_model = base_f_model
        self.remap_func = remap_func

    def forward(self, x0, in_signal, t):
        """
        Args:
            x0 (torch.Tensor): (B, F_s)
            in_signal (torch.Tensor): (B, F_a)
            t (torch.Tensor): (1, )
        return: (B, F_s)
        """
        s = torch.cat([in_signal, x0], dim=-1)
        return self.f_model(s)


class ODEBlock(nn.Module):
    def __init__(self, func: ODEFunc, dt: torch.Tensor, method="euler"):
        super().__init__()
        self.func = func
        self.method = method
        self.register_buffer('dt', dt)
        self.solver = get_solver(self.method, self.func, self.dt)

    def _apply(self, fn):
        super()._apply(fn)
        # Recreate solver with the newly moved dt buffer
        self.solver = get_solver(self.method, self.func, self.dt)
        return self

    def get_f_model(self):
        return self.func.f_model

    def get_next_state(self, action, state, t=None):
        if state.ndim == 1 and action.ndim == 1:
            state = state[None]  # (1, F_s)
            action = action[None]  # (1, F_a)

        assert state.ndim == 2
        assert action.ndim == 2
        assert state.shape[0] == action.shape[0]

        # init_device = state.device
        # state = state.to(self.dt.device)
        # action = action.to(self.dt.device)

        out = self.solver.step(state, action, t)
        # return out.to(init_device)
        return out

    def forward(self, ts, actions, y0):
        """
        Args:
            ts (torch.Tensor): (T, )
            actions (torch.Tensor): (B, F_a, T)
            y0 (torch.Tensor): (B, F_s)
        return: (B, F_s, T)
        """
        # return self.solver.Compute_Dynamics(y0, ts=ts, Input=None)
        # Last action is not used as it is a placeholder
        return odeint_w_in_signal(
            self.func, y0, actions, ts, self.dt, method=self.method
        )


class ODEBlockManualClipping(ODEBlock):
    # Solve an ODE
    def __init__(
        self, func, dt: torch.Tensor, max_lim_vel, min_lim_vel, method="euler"
    ):
        super().__init__(func, dt, method)
        self.odesolver = get_solver(method, self.func, dt)

        # Limit Tensor
        self.max_lim_tensor = nn.Parameter(torch.ones(4), requires_grad=False)
        self.max_lim_tensor *= torch.inf
        self.max_lim_tensor[1] = max_lim_vel

        self.min_lim_tensor = nn.Parameter(torch.ones(4), requires_grad=False)
        self.min_lim_tensor *= -torch.inf
        self.min_lim_tensor[1] = min_lim_vel

    def clip(self, x):
        return torch.clamp(x, self.min_lim_tensor, self.max_lim_tensor)

    def forward(self, ts, in_signal, x0):
        """
        Args:
            ts (torch.Tensor): T
            actions (torch.Tensor): [B, F_a, T]
            y0 (torch.Tensor): [B, F_s]
        """
        # Calculate the whole next trajectory
        # Input: [B, F_a, T] - Sequence of Actions
        # x0: [B, F_s] - Initial Value
        # t0: [B, T] - Length of Time per sample
        T = in_signal.shape[-1]
        X = [x0]
        # The output should not contain the initial conditions, it should start
        # from x1
        if ts.ndim > 1:
            t = ts[:, [0]]
            # t: B, 1
        else:
            t = ts[0]

        for n in range(1, T):
            a_t = in_signal[:, :, n - 1]
            X_old = X[n - 1]
            X_new = self.odesolver.step(X_old, a_t, t)
            X_new = self.clip(X_new)
            X.append(X_new)
            t = t + self.dt
        return torch.stack(X, dim=-1)


class SDEFunc(nn.Module):
    # Drift function f and diffusion function g for the SDE
    def __init__(self, f_model: MLP, g_model: MLP):
        super().__init__()
        self.f_model = f_model
        self.g_model = g_model

    def f(self, x0, in_signal, t):
        """
        This should match the ODEFunc
        Args:
            x0 (torch.Tensor): (B, F_s)
            in_signal (torch.Tensor): (B, F_a)
            t (torch.Tensor): (1, )
        Returns:  (B, F_s)
        """
        s = torch.concatenate([in_signal, x0], dim=-1)
        return self.f_model(s)

    def g(self, x0, in_signal, t):
        s = torch.concatenate([in_signal, x0], dim=-1)
        return self.g_model(s)


class AugSDEFuncDrif(SDEFunc):
    # Augmented MLP as the drift function f of a SDE for target transition dynamics
    def __init__(self, f_model: MLP, g_model: MLP, remap_func: MLP):
        super().__init__(f_model, g_model)
        self.remap_func = remap_func

    def f(self, x0, in_signal, t):
        """
        Args:
            x0 (torch.Tensor): (B, F_s)
            in_signal (torch.Tensor): (B, F_a)
            t (torch.Tensor): (1, )
        return: (B, F_s)
        """
        out = super().f(x0, in_signal, t)
        return self.remap_func(torch.concatenate([out, in_signal], -1))


class SDEBlock(nn.Module):
    # Solve an SDE
    def __init__(self, func: SDEFunc, dt, method="euler", noise_type='diagonal'):
        super().__init__()
        self.func = func
        self.method = method
        self.register_buffer('dt', dt)
        self.noise_type = noise_type

    def get_next_state(self, action, state):
        if state.ndim == 1 and action.ndim == 1:
            state = state[None]  # (1, F_s)
            action = action[None]  # (1, F_a)

        assert state.ndim == 2
        assert action.ndim == 2
        assert state.shape[0] == action.shape[0]

        ts = torch.arange(2, device=self.dt.device) * self.dt  # (2, )
        action = action[..., None].expand(-1, -1, 2)  # (B, F_a, 2)

        next_state = self.forward(ts, action, state)[..., -1]  # (B, F_s)
        return next_state

    def forward(self, ts, actions, y0):
        """
        Args:
            ts (torch.Tensor): (T, )
            actions (torch.Tensor): (B, F_a, T)
            y0 (torch.Tensor): (B, F_s)
        return: (B, F_s, T)
        """
        return sdeint_noise_w_in_signal(
            self.func, y0, actions, ts, self.dt, noise_type=self.noise_type
        )


class LearnedTrans(nn.Module):
    def __init__(self, trained_block: ODEBlock | SDEBlock, mean_value, std_value, f_s):
        super().__init__()
        self.trans_func = trained_block
        self.trans_func.eval()
        self.trans_func.requires_grad_(False)
        self.trans_func.cpu()

        self.state_mean_value = nn.Parameter(mean_value[:f_s], requires_grad=False)
        self.state_std_value = nn.Parameter(std_value[:f_s], requires_grad=False)
        self.action_mean_value = nn.Parameter(mean_value[f_s:], requires_grad=False)
        self.action_std_value = nn.Parameter(std_value[f_s:], requires_grad=False)

    def forward(self, s0, a0):
        """
        one-step transition modelled by one-step forward in ODE/SDE solver
        Args:
            s0: (F_s,) (numpy)
            a0: (F_a,) (numpy)
        Returns:
            s1: (F_s,) (numpy)
        """
        s0 = torch.from_numpy(s0)
        a0 = torch.from_numpy(a0)

        assert s0.shape[0] == self.state_mean_value.shape[0]

        s0 = s0.to(self.state_mean_value.device)
        a0 = a0.to(self.state_mean_value.device)

        s0 = standardise(s0, self.state_mean_value, self.state_std_value)
        a0 = standardise(a0, self.action_mean_value, self.action_std_value)
        out = self.trans_func.get_next_state(action=a0, state=s0).squeeze(dim=0)  # (F_s,)
        out = unstandardise(out, self.state_mean_value, self.state_std_value)

        # Convert back to numpy
        return out.numpy(force=True)


class LearnedTransDiff(nn.Module):
    def __init__(self, trained_block: ODEBlock | SDEBlock, mean_value, std_value, f_s):
        super().__init__()
        self.trans_func = trained_block
        self.trans_func.eval()
        self.trans_func.requires_grad_(False)

        self.state_mean_value = nn.Parameter(mean_value[:f_s].view(1, -1), requires_grad=False)  # (1, F_s)
        self.state_std_value = nn.Parameter(std_value[:f_s].view(1, -1), requires_grad=False)
        self.action_mean_value = nn.Parameter(mean_value[f_s:].view(1, -1), requires_grad=False)  # (1, F_a)
        self.action_std_value = nn.Parameter(std_value[f_s:].view(1, -1), requires_grad=False)

    def forward(self, s0, a0):
        """
        one-step differentiable transition modelled by one-step forward in ODE/SDE solver
        Args:
            s0: (B, F_s) (tensor)
            a0: (B, F_a) (tensor)
        Returns:
            s1: (B, F_s) (tensor)
        """
        s0 = s0.to(self.state_mean_value.device)
        a0 = a0.to(self.state_mean_value.device)

        s0 = standardise(s0, self.state_mean_value, self.state_std_value)
        a0 = standardise(a0, self.action_mean_value, self.action_std_value)

        out = self.trans_func.get_next_state(action=a0, state=s0)  # (B, F_s)
        out = unstandardise(out, self.state_mean_value, self.state_std_value)

        return out


def get_grad_penalty(real_samp, gen_samp, model_d, in_signal, reg_param=10):
    # from: https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/WGAN_GP.py
    batch_size = real_samp.shape[0]
    alpha = torch.rand((batch_size, 1, 1), device=real_samp.device)
    x_hat = alpha * real_samp + (1 - alpha) * gen_samp   # (B, F_s, T)
    pred_hat = model_d(x_hat, in_signal)  # (B, ), critic score for each interpolated sample
    gradients = torch.autograd.grad(
        outputs=pred_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(pred_hat),
        create_graph=True,
        retain_graph=True,
    )[0]  # (B, F_s, T)
    gradients = gradients.reshape([batch_size, -1])
    return reg_param * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

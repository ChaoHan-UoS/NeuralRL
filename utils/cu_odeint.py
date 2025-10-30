import torch
import abc


class _ODESolver(abc.ABC):
    def __init__(self, func, dt: torch.Tensor):
        self.dt = dt
        self.f = func

    @abc.abstractmethod
    def step(self, X, I, t):
        pass

class _RK4(_ODESolver):
    def step(self, xt, at, t):
        k1 = self.f(xt, at, t)
        k2 = self.f(xt + k1 * self.dt / 2, at, t + self.dt / 2)
        k3 = self.f(xt + k2 * self.dt / 2, at, t + self.dt / 2)
        k4 = self.f(xt + k3 * self.dt, at, t + self.dt)
        x_new = xt + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_new

class _RK2(_ODESolver):
    def step(self, xt, at, t):
        k1 = self.f(xt, at, t)
        k2 = self.f(xt + k1 * self.dt, at, t + self.dt)
        x_new = xt + 1 / 2 * (k1 + k2) * self.dt
        return x_new

class _Euler(_ODESolver):
    def step(self, xt, at, t):
        return xt + self.dt * self.f(xt, at, t)


def get_solver(method: str, func, dt):
    params = (func, dt)
    if method == "euler":
        odesolver = _Euler(*params)
    elif method == "rk4":
        odesolver = _RK4(*params)
    elif method == "rk2":
        odesolver = _RK2(*params)
    else:
        raise NotImplementedError(
            f"ODESolver: '{method}' has not been implemented. Please try: ['euler', 'rk4', 'rk2']"
        )
    return odesolver


def odeint_w_in_signal(
    func, # ODEFunc
    x0: torch.Tensor,  # B, F_s
    in_signal: torch.Tensor,  # B, F_a, T
    ts: torch.Tensor,  # T
    dt: torch.Tensor,
    method="euler",
    use_all_actions=False
):
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device and dt.device == ts.device
    ), f"""x0, func, dt and ts should be on the same device, but got: x0 on {x0.device}, 
           dt on {dt.device} and ts on {ts.device}!"""

    params = (func, dt)
    if method == "euler":
        odesolver = _Euler(*params)
    elif method == "rk4":
        odesolver = _RK4(*params)
    elif method == "rk2":
        odesolver = _RK2(*params)
    else:
        raise NotImplementedError(
            f"""ODESolver: '{method}' has not been implemented. 
                Please try: ['euler', 'rk4', 'rk2','semi-implicit euler']"""
        )

    T = in_signal.shape[-1]
    # Uses all actions in the input signal if use_all_actions=True,
    # otherwise uses all actions except the last one
    if use_all_actions:
        T += 1

    if ts.ndim > 1:
        t = ts[:, [0]]  # t: B, 1
    else:
        t = ts[0]
    X = [x0]

    for n in range(1, T):
        a_t = in_signal[:, :, n - 1]
        X_old = X[n - 1]
        X_new = odesolver.step(X_old, a_t, t)
        X.append(X_new)
        t = t + dt
    # B, F_s, T
    return torch.stack(X, dim=-1)


def odeint(func, x0: torch.Tensor, ts: torch.Tensor, dt: torch.Tensor, method="euler"):
    """
    Implementation of the ODESolver without the input signal.

    Args:
        func (_type_): NeuralODE of the form: f(X, t, I)
        x0 (torch.Tensor): Starting state
        ts (torch.Tensor): Time points
        dt (torch.Tensor): Time Step
        method (str, optional): ODE Numerical Solver. Defaults to 'euler'.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    # Function should be f(X, t, I)
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device and dt.device == ts.device
    ), f"x0, func, dt and ts should be on the same device, but got: x0 on {x0.device}, dt on {dt.device} and ts on {ts.device}!"

    params = (func, dt)
    if method == "euler":
        odesolver = _Euler(*params)
    elif method == "rk4":
        odesolver = _RK4(*params)
    elif method == "rk2":
        odesolver = _RK2(*params)
    else:
        raise NotImplementedError(
            f"ODESolver: '{method}' has not been implemented. Please try: ['euler', 'rk4', 'rk2']"
        )

    # Calculate the whole next trajectory
    # Input: [B, F_a, T] - Sequence of Actions
    # x0: [B, F_s] - Initial Value
    # t0: [B, T] - Length of Time per sample
    T = ts.shape[0]
    # batch_size=x0.size()[0]
    # N = x0.size()[1]
    # X = torch.zeros([batch_size, N, T], device=x0.device)
    # X[..., 0] = x0
    X = [x0]
    # The output should not contain the initial conditions, it should start
    # from x1
    t = ts[0]

    for n in range(1, T):
        X_old = X[n - 1]
        X_new = odesolver.step(X_old, None, t)
        X.append(X_new)
        t = t + dt
    return torch.stack(X, dim=-1)


def cu_sdeint(
    func, x0: torch.Tensor, ts: torch.Tensor, dt: torch.Tensor, method="euler"
):
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device and dt.device == ts.device
    ), f"x0, func, dt and ts should be on the same device, but got: x0 on {x0.device}, dt on {dt.device} and ts on {ts.device}!"

    # Calculate the whole next trajectory
    # Input: [B, F_a, T] - Sequence of Actions
    # x0: [B, F_s] - Initial Value
    # t0: [B, T] - Length of Time per sample
    T = ts.shape[0]
    # batch_size=x0.size()[0]
    # N = x0.size()[1]
    # X = torch.zeros([batch_size, N, T], device=x0.device)
    # X[..., 0] = x0
    X = [x0]
    # The output should not contain the initial conditions, it should start
    # from x1
    t = ts[0]

    diff_func = func.g
    drift_func = func.f
    dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(dt))
    # Only implement diagonal noise type, i.e. there are F_s independent Browmian Motions
    # diff_func will have output: B, F_s
    noise_shape = (x0.shape[0], x0.shape[-1])

    for n in range(1, T):
        # I = Input[:, :, n-1]
        X_old = X[n - 1]
        # X_new = solver(X_old, I, t)
        # X_new = X_old + drift_func(X_old, None, t) * dt + diff_func(X_old, None, t) @ torch.diag(dw_dist.sample(noise_shape).to(dt.device), diagonal=0)
        X_new = (
            X_old
            + drift_func(X_old, None, t) * dt
            + diff_func(X_old, None, t) * dw_dist.sample(noise_shape).to(dt.device)
        )
        # print(X_new)
        # X[:, :, n] = X_new
        X.append(X_new)
        t = t + dt
    return torch.stack(X, dim=-1)


def cu_sdeint_noise(
    func, x0: torch.Tensor, ts: torch.Tensor, dt: torch.Tensor, method="euler"
):
    """
    Same as cu_sdeint but also looks for a z_function that maps the noise to the initial starting point
    """
    # if x0 is noise, forget the first step and generate everything
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device and dt.device == ts.device
    ), f"x0, func, dt and ts should be on the same device, but got: x0 on {x0.device}, dt on {dt.device} and ts on {ts.device}!"

    # z function used to map the noise to something
    z_func = func.z

    # Calculate the whole next trajectory
    # Input: [B, F_a, T] - Sequence of Actions
    # x0: [B, F_s] - Initial Value
    # t0: [B, T] - Length of Time per sample
    T = ts.shape[0]
    # batch_size=x0.size()[0]
    # N = x0.size()[1]
    # X = torch.zeros([batch_size, N, T], device=x0.device)
    # X[..., 0] = x0
    X = [z_func(x0)]
    # The output should not contain the initial conditions, it should start
    # from x1
    t = ts[0]

    diff_func = func.g
    drift_func = func.f

    dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(dt))
    # Only implement diagonal noise type, i.e. there are F_s independent Browmian Motions
    # diff_func will have output: B, F_s
    noise_shape = (x0.shape[0], x0.shape[-1])

    for n in range(1, T):
        # I = Input[:, :, n-1]
        X_old = X[n - 1]
        # X_new = solver(X_old, I, t)
        # X_new = X_old + drift_func(X_old, None, t) * dt + diff_func(X_old, None, t) @ torch.diag(dw_dist.sample(noise_shape).to(dt.device), diagonal=0)
        X_new = (
            X_old
            + drift_func(X_old, None, t) * dt
            + diff_func(X_old, None, t) * dw_dist.sample(noise_shape).to(dt.device)
        )
        # print(X_new)
        # X[:, :, n] = X_new
        X.append(X_new)
        t = t + dt
    return torch.stack(X, dim=-1)


def sdeint_noise_w_in_signal(
    func, # SDEFunc,
    x0: torch.Tensor,  # (B, F_s)
    in_signal: torch.Tensor,  # (B, F_a, T)
    ts: torch.Tensor,  # (T, )
    dt: torch.Tensor,
    noise_type="diagonal",
):
    # Calculate the whole state trajectory, including the initial state
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device
        and dt.device == ts.device
        and in_signal.device == dt.device
    ), f"x0, func, in_signal, dt and ts should be on the same device, but got: x0 on {x0.device}, \
         dt on {dt.device}, ts on {ts.device}, in_signal on {in_signal}!"

    T = in_signal.shape[-1]
    t = ts[0]
    X = [x0]

    drift_func = func.f
    diff_func = func.g

    dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(dt))
    # Only implement diagonal noise type, i.e. there are F_s independent Brownian Motions
    # diff_func will have output: B, F_s
    if noise_type == "diagonal":
        # One Brownian motion for each state
        noise_shape = (x0.shape[0], x0.shape[-1])  # (B, F_s)
    elif noise_type == "scalar":
        # One Brownian motion is shared across all states
        noise_shape = (x0.shape[0], 1)  # (B, 1)

    for n in range(1, T):
        a_t = in_signal[..., n - 1]  # uses all actions except the last one
        X_old = X[n - 1]
        X_new = (
            X_old + drift_func(X_old, a_t, t) * dt
            + diff_func(X_old, a_t, t) * dw_dist.sample(noise_shape).to(dt.device)
        )  # Euler–Maruyama method
        X.append(X_new)
        t = t + dt
    # (B, F_s, T)
    return torch.stack(X, dim=-1)


def get_noise_shape(x0, noise_type):
    if noise_type == "diagonal":
        # One Browmian motion for each state
        noise_shape = (x0.shape[0], x0.shape[-1])
    elif noise_type == "scalar":
        # One Browmian motion is shared across all states
        noise_shape = (x0.shape[0], 1)
    return noise_shape


def cu_odeint_noise_w_in_signal(
    func,
    x0: torch.Tensor,
    in_signal: torch.Tensor,
    ts: torch.Tensor,
    dt: torch.Tensor,
    method="euler",
):
    # in_signal/ Action [B, F_a, T]
    # if x0 is noise, forget the first step and generate everything
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device
        and dt.device == ts.device
        and in_signal.device == dt.device
    ), f"x0, func, in_signal, dt and ts should be on the same device, but got: x0 on {x0.device}, dt on {dt.device}, ts on {ts.device}, in_signal on {in_signal}!"

    # z function used to map the noise to something
    z_func = func.z

    # Calculate the whole next trajectory
    # Input: [B, F_a, T] - Sequence of Actions
    # x0: [B, F_s] - Initial Value
    # t0: [B, T] - Length of Time per sample
    T = in_signal.shape[-1]
    # batch_size=x0.size()[0]
    # N = x0.size()[1]
    # X = torch.zeros([batch_size, N, T], device=x0.device)
    # X[..., 0] = x0
    # The output should not contain the initial conditions, it should start
    # from x1
    t = ts[0]
    # X = [z_func(x0, t, in_signal[0])]
    X = [z_func(x0)]

    # diff_func = func.g
    drift_func = func.f

    # dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(dt))
    # Only implement diagonal noise type, i.e. there are F_s independent Browmian Motions
    # diff_func will have output: B, F_s
    # noise_shape = (x0.shape[0], x0.shape[-1])

    for n in range(1, T):
        # I = Input[:, :, n-1]
        a_t = in_signal[..., n - 1]
        X_old = X[n - 1]
        # X_new = solver(X_old, I, t)
        # X_new = X_old + drift_func(X_old, None, t) * dt + diff_func(X_old, None, t) @ torch.diag(dw_dist.sample(noise_shape).to(dt.device), diagonal=0)
        X_new = X_old + drift_func(X_old, a_t, t) * dt
        # print(X_new)
        # X[:, :, n] = X_new
        X.append(X_new)
        t = t + dt
    return torch.stack(X, dim=-1)


def sdeint_new(
    func,
    x0: torch.Tensor,
    in_signal: torch.Tensor,
    ts: torch.Tensor,
    dt: torch.Tensor,
    noise: torch.Tensor,
    method="euler",
    noise_type="diagonal",
):
    """
    G_theta approximates g(.,.)Delta W_t instead of just g
    """
    # in_signal/ Action [B, F_a, T]
    # if x0 is noise, forget the first step and generate everything
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device
        and dt.device == ts.device
        and in_signal.device == dt.device
    ), f"x0, func, in_signal, dt and ts should be on the same device, but got: x0 on {x0.device}, dt on {dt.device}, ts on {ts.device}, in_signal on {in_signal}!"

    # Calculate the whole next trajectory
    # Input: [B, F_a, T] - Sequence of Actions
    # x0: [B, F_s] - Initial Value
    # t0: [B, T] - Length of Time per sample
    T = in_signal.shape[-1]
    X = [x0]
    t = ts[0]
    diff_func = func.g
    drift_func = func.f

    # dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(dt))
    # Only implement diagonal noise type, i.e. there are F_s independent Browmian Motions
    # diff_func will have output: B, F_s
    if noise_type == "diagonal":
        # One Browmian motion for each state
        noise_shape = (x0.shape[0], x0.shape[-1])
    elif noise_type == "scalar":
        # One Browmian motion is shared across all states
        noise_shape = (x0.shape[0], 1)

    for n in range(1, T):
        # I = Input[:, :, n-1]
        a_t = in_signal[..., n - 1]
        X_old = X[n - 1]
        X_new = X_old + drift_func(X_old, a_t, t) * dt + diff_func(X_old, a_t, t, noise)
        X.append(X_new)
        t = t + dt
    return torch.stack(X, dim=-1)


def residual_sdeint(
    func,
    x0: torch.Tensor,
    in_signal: torch.Tensor,
    ts: torch.Tensor,
    dt: torch.Tensor,
    noise: torch.Tensor,
    method="euler",
    noise_type="diagonal",
):
    """
    G_theta approximates g*DW_t instead of just g
    """
    # Assumes that the diff parameter learns g(.,.)\Delta W_t
    # in_signal/ Action [B, F_a, T]
    # if x0 is noise, forget the first step and generate everything
    # Check dt type
    assert isinstance(dt, torch.Tensor), f"dt must be a tensor, but is now {dt.dtype}"
    # Check device
    assert (
        x0.device == dt.device
        and dt.device == ts.device
        and in_signal.device == dt.device
    ), f"x0, func, in_signal, dt and ts should be on the same device, but got: x0 on {x0.device}, dt on {dt.device}, ts on {ts.device}, in_signal on {in_signal}!"

    # Calculate the whole next trajectory
    # Input: [B, F_a, T] - Sequence of Actions
    # x0: [B, F_s] - Initial Value
    # t0: [B, T] - Length of Time per sample
    T = in_signal.shape[-1]
    X = [x0]
    t = ts[0]
    diff_func = func.g
    drift_func = func.f

    # dw_dist = torch.distributions.Normal(loc=0, scale=torch.sqrt(dt))
    # Only implement diagonal noise type, i.e. there are F_s independent Browmian Motions
    # diff_func will have output: B, F_s
    if noise_type == "diagonal":
        # One Browmian motion for each state
        noise_shape = (x0.shape[0], x0.shape[-1])
    elif noise_type == "scalar":
        # One Browmian motion is shared across all states
        noise_shape = (x0.shape[0], 1)

    for n in range(1, T):
        # I = Input[:, :, n-1]
        a_t = in_signal[..., n - 1]
        X_old = X[n - 1]
        X_new = X_old + drift_func(X_old, a_t, t) * dt
        # Diffusion Function now models around the ODE and is not a separate mapping
        X_new += diff_func(X_new, a_t, t, noise)
        X.append(X_new)
        t = t + dt
    return torch.stack(X, dim=-1)

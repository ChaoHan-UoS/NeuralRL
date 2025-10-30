import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from torchdiffeq import odeint
device = "cuda"


class EncoderRNN(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, pack_input=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.pack_input = pack_input

    def forward(self, ats, sts, lens=None):
        # ats: (B, F_a, T - 1), dropped the last action/state
        # sts: (B, F_s, T - 1)
        # `pack_input=True` requires `lens` to be provided
        assert not self.pack_input or lens is not None
        assert ats.shape[-1] == sts.shape[-1]
        aug_sts = torch.concatenate([ats, sts], dim=1)
        aug_sts = aug_sts.transpose(-1, -2)  # (B, T - 1, F_s + F_a)
        if self.pack_input:
            aug_sts = pack_padded_sequence(
                aug_sts,
                lens - 1,           # drop the last element
                batch_first=True,
                enforce_sorted=False
            )

        # (B, Latent_Dim)
        return self.gru(aug_sts)[1][-1]


class LatentODE(nn.Module):
    def __init__(
        self,
        encoder: EncoderRNN,
        f_s,
        f_a,
        dt_value,
        enc_output,
        latent_dim,
        encoder_to_hidden=20,
        decoder_mlp_size=100,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.register_buffer('timesteps', (torch.arange(2) * dt_value))

        # Encoder ouput -> mean and log variance of the initial latent z_0
        self.hidden_to_z0 = nn.Sequential(
            nn.Linear(enc_output, encoder_to_hidden),
            nn.Tanh(),
            nn.Linear(encoder_to_hidden, 2 * latent_dim),
        )

        # Optional projection to speed up high-dimensional state computations
        # and enable torchdiff integration
        # z_i, a_i, o_i -> transformed z_i
        self.aug_state_2_decoder = nn.Linear(latent_dim + f_a + f_s, latent_dim)

        # Derivative func f_ode of the latent ODE decoder
        # transformed z_i -> f_ode(transformed z_i)
        self.decoder_odefunc = nn.Sequential(
            nn.Linear(latent_dim, decoder_mlp_size),
            nn.Tanh(),
            nn.Linear(decoder_mlp_size, decoder_mlp_size),
            nn.Tanh(),
            nn.Linear(decoder_mlp_size, latent_dim)
        )

        # Latent ODE decoder solver
        self.decoder = DiffeqSolver(
            ode_func=lambda t, y: self.decoder_odefunc(y),
            method='rk4',
            odeint_rtol=1e-5,
            odeint_atol=1e-6
        )

        # Map z_i -> s_i
        self.latent_2_state_proj = nn.Linear(self.latent_dim, f_s)
        # self.latent_2_state_proj = nn.Sequential(
        #     nn.Linear(self.latent_dim, 25),
        #     nn.LeakyReLU(),
        #     nn.Linear(25, f_s)
        # )

    def encode(self, actions, states, lens=None):
        """
        Encode a batch of actions and states to the stats for dist of
        initial latent z_0:
        - actions: (B, F_a, T - 1)
        - states:  (B, F_s, T - 1)
        """
        out = self.encoder(actions, states, lens=lens)  # (B, Latent_Dims)
        out = self.hidden_to_z0(out)                    # (B, 2 * Latent_Dim)

        mu = out[:, :self.latent_dim]                   # (B, Latent_Dim)
        log_var = out[:, self.latent_dim:]
        return mu, log_var

    @torch.no_grad()
    def get_init_latent(self, batch_size=1):
        # Sample z0 from Standard Gaussian prior
        return torch.randn((batch_size, self.latent_dim))

    def sample_latent(self, mu, log_var):
        dist = torch.distributions.Normal(mu, torch.exp(log_var * 0.5))  # N(μ, σ^2)
        # Sample got by reparam trick y = μ + σ·ε: (B, Latent_Dim)
        return dist.rsample()

    def get_next_states(self, actions, z0, s0, tf=False):
        """
        Autoregressively roll out T -1 steps:
        - actions: (B, F_a, T - 1), a_0, ..., a_{T-2}
        - z0:      (B, latent_dim)
        - s0:      (B, F_s, T) if tf=True, else (B, F_s)
        Returns:   (B, F_s, T - 1) of predicted next states s_1, ..., s_{T-1}
        """
        assert s0.ndim == (3 if tf else 2)
        Tm1 = actions.shape[-1]
        next_states = []

        for n in range(Tm1):
            # 1) Pick the current state s_i
            if tf:
                s_curr = s0[..., n]
            elif n == 0:
                s_curr = s0
            else:
                s_curr = next_states[-1]

            # 2) Compute next latent: z_i, a_i, s_i -> z_{i+1}
            z0 = self.get_next_latent(actions[..., n], z0, s_curr)

            # 3) Map z_{i+1} -> s_{i+1}
            s_next = self.latent_2_state_proj(z0)
            next_states.append(s_next)

        return torch.stack(next_states, dim=-1)

    def get_next_latent(self, at, zt, st):
        """
        - at: (B, F_a)
        - zt: (B, latent_dim)
        - st: (B, F_s)
        Returns: predicted next latent (B, Latent_Dim)
        """
        assert at.ndim == zt.ndim == st.ndim == 2
        assert zt.shape[-1] == self.latent_dim

        s_aug = torch.concatenate([at, zt, st], dim=-1)
        s_aug = self.aug_state_2_decoder(s_aug)  # transformed z_i

        # Solve for two time‐points [0, dt] with s_aug at t=0
        next_z = self.decoder(s_aug, self.timesteps)  # (B, 2, Latent_Dim)

        return next_z[:, -1]


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol, odeint_atol):
        super(DiffeqSolver, self).__init__()
        self.ode_func = ode_func
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps, odeint_rtol=None, odeint_atol=None, method=None):
        """
        Decode a latent trajectory by solving the ODE:
        - first_point: (B, D), the state at t=0
        - time_steps:  (T, ), time grid (e.g. [0, dt])
        Returns:       (B, T, D), predicted latent traj
        """
        if not odeint_rtol:
            odeint_rtol = self.odeint_rtol
        if not odeint_atol:
            odeint_atol = self.odeint_atol
        if not method:
            method = self.ode_method

        # Call the ODE integrator (from torchdiffeq)
        pred = odeint(
            self.ode_func,
            first_point,
            time_steps,
            rtol=odeint_rtol,
            atol=odeint_atol,
            method=method
        )  # (T, B, D)
        pred = pred.permute(1, 0, 2)  # (B, T, D)

        return pred
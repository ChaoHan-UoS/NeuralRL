General Comments:
- Any file that has "_simple", trains the model only once and then freezes it. This worked fine for the Swimmer, but not for the Hopper. I speculate that this is because there is a big change in the policy of the agent because of the unhealthy regions (early termination of the environment due to the robot reaching an unhealthy state), and thus what the model learns initially is not enough. Any file that has "_cont" trains the model and the agent together, as in the pseudocode of the report.
- "--e" parameter is set high for pomdps and low for fully-observable environments

# Agents
I used Soft-Actor Critic throught - url{(https://spinningup.openai.com/en/latest/algorithms/sac.html). The implementation under `sac/` is from https://github.com/pranz24/pytorch-soft-actor-critic/tree/master .

In the beginning I also experimented with Actor-Critic (just for deterministic Cartpole). This can be found in `integ_model_learning_2 _actor_critic.ipynb`, and `testing_learned_env.ipynb`

# Cartpole
Cartpole is trained in a sample-based way (i.e. train the model at the start freeze it and train the agent). The `train_agent` function used to train the agent of the cartpole counts number of episodes instead of steps as in the mujoco environment.

## Approach 1: (wgan with two sources of noise) - no longer using:
- model-free (stoch): `run_model_free_cartpole_stoch.py`
- model-free (deterministic): `run_model_free_cartpole.py`
- mb-ode: `run_mbnode_cartpole.py`
- mbsde: `run_mbnode_cartpole_stoch.py`


## Approach 2: (wgan with just one source of noise) - use this:
`run_mbsde_cartpole_stoch_remove_noise.py`
Cartpole links:
- https://wandb.ai/stefanosio/continuous-cartpole
- https://wandb.ai/stefanosio/continuous-cartpole-sde
- https://wandb.ai/stefanosio/rl-cartpole-sde-redo

## Other:
Cartpole with Data Augmentation found in `continous_cartpole_stoch_adaptation.ipynb `. Data augmentation is implemented similar to https://arxiv.org/pdf/2102.04764


# Mujoco Tasks (Fully-Observable)
## Stoch-Swimmer
Introducing independent Gaussian noise to the stiffness parameter of one of the joints in the Swimmer

Simple-* https://wandb.ai/stefanosio/mpc-stiff-swimmer-new-runs

Latent-* https://wandb.ai/stefanosio/mpc-stiff-swimmer-cont-latent-comparison

### Models
- Simple (NeuralODE)
```bash
python mpc_simple_cont.py --type mbsde --env swimmer --batch-size 128 --init-data 1000 --k 1500 --alpha 0.2 --min-q --h 10 --e 1
```
- Simple (NeuralSDE)
```bash
python mpc_simple_cont.py --type mbode --env swimmer --batch-size 128 --init-data 1000 --k 1500 --alpha 0.2 --min-q --h 10 --e 1
```
- Latent-ODE
```bash
mpc_latent_cont.py --env swimmer --init-data 1000 --alpha 0.2 --k 1000 --type mbode
```
- Latent-SDE *
```bash
mpc_latent_cont.py --env swimmer --init-data 1000 --alpha 0.2 --k 1000 --type mbsde
```


## Stoch-Hopper
Adding independent Gaussian noise to the wind parameter of the hopper. Initially the first environment on hopper had noise drawn from a skewed distribution. Because this was very hard to optimize, I changed the distribution to Gaussian

Simple-* https://wandb.ai/stefanosio/mpc-hopper-gauss-stoch-wind-new-runs

Latent-* https://wandb.ai/stefanosio/mpc-hopper-gauss-stoch-wind-cont-latent-comparison

### Models
- Simple-NeuralODE

```bash
python mpc_simple_cont.py --type mbode --env hopper-gaus --alpha 0.25 --k 1700 --batch-size 128 --init-data 1000 --e 1 --lr 0.0003 --h 15 --min-q
```

- Simple-NeuralSDE 

```bash
python mpc_simple_cont.py --type mbsde --env hopper-gaus --alpha 0.25 --k 1700 --batch-size 128 --init-data 1000 --e 1 --lr 0.0003 --h 15 --min-q
```

- Latent-ODE
```bash
python mpc_latent_cont.py --env hopper-gaus --init-data 1000 --alpha 0.25 --k 1700 --type mbode
```
- Latent-SDE
```bash
python mpc_latent_cont.py --env hopper-gaus --init-data 1000 --alpha 0.25 --k 1700 --type mbsde
```

# Mujoco Tasks (Partially-Observable)
## Latent-Stoch-Swimmer (No Position)
Introducing independent Gaussian noise to the stiffness parameter of one of the joints in the Swimmer

- No Position: https://wandb.ai/stefanosio/mpc-pomdp-swimmer-no-pos-cont-latent-comparison
- No Velocity: https://wandb.ai/stefanosio/mpc-pomdp-swimmer-no-vel-cont-latent-comparison

### Models
- Latent-ODE
- Latent-SDE


## Latent-Stoch-Hopper
Adding independent Gaussian noise to the wind parameter of the hopper. Initially the first environment on hopper had noise drawn from a skewed distribution. Because this was very hard to optimize, I changed the distribution to Gaussian
- No Position: https://wandb.ai/stefanosio/mpc-pompd-hopper-no-pos-cont-latent-comparison
- No Velocity: https://wandb.ai/stefanosio/mpc-pompd-hopper-no-vel-cont-latent-comparison

(* All this experiments need more than 8 hours to complete on bessemer - special permission should be requested on slurm)

### Models
- Latent-ODE
    - No Velocity:
    ```bash
    python mpc_latent_cont.py --env pomdp-hopper-no-vel --init-data 25000 --alpha 0.25 --k 1700 --type mbode --use-latent --e 20 --latent-dim 150 --min-q
    ```
    - No Position:
    ```bash
    python mpc_latent_cont.py --env pomdp-hopper-no-pos --init-data 25000 --alpha 0.25 --k 1700 --type mbode --use-latent --e 20 --latent-dim 150 --min-q
    ```
- Latent-SDE
    - No Velocity:
    ```bash
    python mpc_latent_cont.py --env pomdp-hopper-no-vel --init-data 25000 --alpha 0.25 --k 1700 --type mbsde --use-latent --e 20 --latent-dim 150 --min-q
    ```
    - No Position:
    ```bash
    python mpc_latent_cont.py --env pomdp-hopper-no-pos --init-data 25000 --alpha 0.25 --k 1700 --type mbsde --use-latent --e 20 --latent-dim 150 --min-q
    ```

*This should probably need be the name of this model, as the name is already taken.

Why is it possible for ODEs to reach the same performance as SDE:
I speculate that this can happen, if noise is to an environment that can otherwise be simulated by the actions of a random policy. In the case of stoch-swimmer, I think certain actions of the policy can simulate trajectories that look like the swimmer's joint is stiff with different intensities, so I expect that given enough exploration, NeuralODE+SAC can eventually converge to the max expected reward. Although this might be slower. It may also be the case, that the noise is not enough.

# Important Files:
Contains ode/sde solvers: `cu_odeint.py`

# Adaptation:
`cartpole_policy_shift.py`, `cartpole_policy_shift.ipynb` - cartpole adaptation without training the policy (through policy shifting)
`cartpole_policy_shift_sde.ipynb` - cartpole adaptation without training the policy with actuator noise.

# Plotting:
`plot_cartpole.ipynb` contains code to plot the boxplots of the stoch-cartpole environment
`plot_cartpole_adaptation.ipynb` contains code to plot the boxplots for the cartpole adaptation experiments 


# Other files - files no longer using, have various techniques.
ODE with delays: `train_delayed_ode.ipynb`

ODE-RNN from https://arxiv.org/abs/2006.16210:
- `train_latent_ode_w_rnn_cartpole.ipynb`
- `train_latent_ode_w_rnn_fixed_len.ipynb`

Sample-Based MPC: `mpc.py` , `mpc.ipynb`

Double Well: all files that include dw in their name (was debugging wgans on simple benchmark)

Ideas about cartpole environments (abandoned): `custom_env.py`

Adaptation using data augmentation: `continous_cartpole_stoch_data_augmentation.ipynb`, `continous_cartpole_stoch_data_augmentation_only.ipynb `

Experimenting with continous cartpole: `continous_cartpole_stoch.ipynb`, `continous_cartpole.ipynb`

Initial experiments on policy shifting: `cartpole_policy_shift_initial.ipynb`
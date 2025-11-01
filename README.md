# Neural ODE and SDE Models for Adaptation and Planning in Model-Based Reinforcement Learning

This repository contains the PyTorch implementation of the paper "[Neural ODE and SDE Models for Adaptation and Planning in Model-Based Reinforcement Learning](https://openreview.net/pdf?id=T6OrPlyPV4)".

The paper systematically studies neural ODE/SDE models and their latent variants for model-based policy learning and adaptation in continuous control tasks, highlighting the advantages of policy planning over latent SDE in scenarios with noisy transition dynamics and partial observability.

## Installation

We recommend using conda for package management. The required Python dependencies are specified in `requirements.yml`. An NVIDIA GPU is recommended for faster training. We also recommend [Weights & Biases](https://wandb.ai/site/) for optional experiment tracking and visualization. 

Clone this repository and set up a new conda environment:

```bash
$ conda env create -f requirements.yml
$ conda activate neural_rl
```

## Running Experiments

### Model-based Policy Adaptation

We evaluate policy adaptation via inverse dynamics in both deterministic and stochastic CartPole environments with increasing pole lengths. We compare different policy baselines including non-adapted, scratch-trained, and adapted policies using neural ODE, neural SDE, and Gaussian ensemble networks as transition models. 

The non-adapted and adapted policies can be run via:

```bash
$ python cartp_varlen.py --env ${env_type} --length ${target_pole_len} --model ${model_type} --policy-ckpt ${source_policy_ckpt} --model-ckpt ${source_model_ckpt}
```

where:
- `env_type` can be `stoch` or `det`
- `target_pole_len` is the pole length in target environments ranging from `1.0` to `3.8` with step size `0.2`
- `model_type` can be `ode`, `sde`, or `ens`
- `source_policy_ckpt` and `source_model_ckpt` are the checkpoints of the pre-trained policy and transition model on the source environment with pole length `1.0`

These checkpoints can be obtained by running:
- Policy checkpoint: `$ python cartp.py --env ${env_type} --model mf`
- Model checkpoint: `$ python cartp.py --env ${env_type} --model ${model_type} --no-train-rl` (where `--model` is `ode`, `sde`, or `ens`)

The scratch-trained policy baseline can be run via:

```bash
$ python cartp_varlen_mf.py --env ${env_type} --length ${target_pole_len}
```

where `env_type` and `target_pole_len` are defined as above.

### Model-based Policy Learning

We compare policies trained in real and learned CartPole environments using neural ODE and SDE as transition models. The experiments can be run via:

```bash
$ python cartp.py --env ${env_type} --model ${model_type}
```

where `env_type` can be `stoch` or `det`; `model_type` can be `mf`, `ode`, `sde`, or `ens`.

We further compare model-free and model-based policies in MuJoCo tasks with stochastic transition dynamics and partial observability. These policy baselines include model-free SAC, MBPO, and policies planned via MPC using neural/latent ODE/SDE.

#### Stochastic Swimmer

```bash
# SAC
$ python mpc_latent.py --env ${env_name} --type mf --k 1000 --alpha 0.2

# MBPO
$ python mpc_latent.py --env ${env_name} --type mbpo

# Neural ODE
$ python mpc_latent.py --env ${env_name} --type mbode --k 1000 --alpha 0.2

# Neural SDE
$ python mpc_latent.py --env ${env_name} --type mbsde --k 1000 --alpha 0.2

# Latent ODE
$ python mpc_latent.py --env ${env_name} --type mbode --k 1000 --alpha 0.2 --latent-m

# Latent SDE
$ python mpc_latent.py --env ${env_name} --type mbsde --k 1000 --alpha 0.2 --latent-m
```

where `${env_name}` can be `swimmer`, `pomdp-swimmer-no-pos`, or `pomdp-swimmer-no-vel`.

#### Stochastic Hopper and Walker2d

```bash
# SAC
$ python mpc_latent.py --env ${env_name} --type mf --k 1700 --alpha 0.25

# MBPO
$ python mpc_latent.py --env ${env_name} --type mbpo

# Neural ODE
$ python mpc_latent.py --env ${env_name} --type mbode --k 1700 --alpha 0.25

# Neural SDE
$ python mpc_latent.py --env ${env_name} --type mbsde --k 1700 --alpha 0.25

# Latent ODE
$ python mpc_latent.py --env ${env_name} --type mbode --k 1700 --alpha 0.25 --latent-m

# Latent SDE
$ python mpc_latent.py --env ${env_name} --type mbsde --k 1700 --alpha 0.25 --latent-m
```

where `${env_name}` can be `hopper`, `pomdp-hopper-no-pos`, `pomdp-hopper-no-vel`, `walker2d`, `pomdp-walker2d-no-pos`, or `pomdp-walker2d-no-vel`.

## Acknowledgements

The code implementation is inspired by and partially based on the following repositories:

- [mbrl-smdp-ode](https://github.com/dtak/mbrl-smdp-ode)
- [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic)
- [mbpo_pytorch](https://github.com/Xingyu-Lin/mbpo_pytorch)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{han2025neural,
    title={Neural {ODE} and {SDE} Models for Adaptation and Planning in Model-Based Reinforcement Learning},
    author={Chao Han and Stefanos Ioannou and Luca Manneschi and T.J. Hayward and Michael Mangan and Aditya Gilra and Eleni Vasilaki},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=T6OrPlyPV4},
}
```

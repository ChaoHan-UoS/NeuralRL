from argparse import ArgumentParser
from uuid import uuid4

import torch
import torch.nn as nn
import wandb

from env.swimmer import (
    SwimmerStochStiffnessZeroMean, POMDPSwimmerNoPosition, POMDPSwimmerNoVelocity
)
from env.hopper import (
    HopperStochWindGaussian, POMDPHopperNoPosition, POMDPHopperNoVelocity
)
from env.walker2d import (
    Walker2dWindGaussian, POMDPWalker2dNoPosition, POMDPWalker2dNoVelocity
)
from sac.sac import SAC
from utils.utils import MLP
from utils.utils_models import ODEFunc, ODEBlock
from utils.mpc import EncoderRNN, LatentODE
from utils.mpc_utils import (
    MBRL, ReplayMemory, TrajectoryPolicy,Transition, TransitionLatent,
    SDEFuncLatent, SDEBlockLatent, CollapseDiscWLayerNorm, EarlyStopper
)


device = "cuda"
ENV_DICT = {
    'swimmer': SwimmerStochStiffnessZeroMean,
    'hopper-gaus': HopperStochWindGaussian,
    'walker2d': Walker2dWindGaussian,
    'pomdp-swimmer-no-pos': POMDPSwimmerNoPosition,
    'pomdp-swimmer-no-vel': POMDPSwimmerNoVelocity,
    'pomdp-hopper-no-pos': POMDPHopperNoPosition,
    'pomdp-hopper-no-vel': POMDPHopperNoVelocity,
    'pomdp-walker2d-no-pos': POMDPWalker2dNoPosition,
    'pomdp-walker2d-no-vel': POMDPWalker2dNoVelocity,
}
ENV_PARAMS = {'swimmer':dict(std=500)}

parser = ArgumentParser()
parser.add_argument('--env', choices=ENV_DICT.keys(), default='swimmer',
                    help="Training environment, default is 'swimmer'.")
parser.add_argument('--type', choices=('mbsde', 'mbode', 'mf', 'mbpo'), default='mbsde',
                    help="Model type: 'mbsde' (default), 'mbode', or 'mf'.")
parser.add_argument('--noise-type', choices=('diagonal', 'scalar'), default='scalar',
                    help="Noise type: 'diagonal' or 'scalar' (default: 'scalar').")
parser.add_argument('--cut-len', default=None, type=int,
                    help="Cutoff length for trajectories (default: None).")
parser.add_argument('--init-data', default=15000, type=int,
                    help="Initial env steps for collecting training data (default: 15000).")
parser.add_argument('--init-data-val', default=10000, type=int,
                    help="Initial env steps for collecting validation data (default: 10000).")
parser.add_argument('--k', default=1000, type=int,
                    help="Population size/rollouts in MPC (default: 1000).")
parser.add_argument('--h', default=10, type=int,
                    help="Prediction horizon for MPC (default: 10).")
parser.add_argument('--e', default=1, type=int,
                    help="Number of epochs per training round (default: 1).")
parser.add_argument('--min-q', default=False, action='store_true',
                    help="Take the min Q-value during MPC.")
parser.add_argument('--model_batch_size', default=128, type=int,
                    help="Batch size for model training (default: 128).")

parser.add_argument('--replay_size', default=1000000, type=int,
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--policy_hidden_size', default=256, type=int,
                    help='hidden size for policy networks (default: 256)')
parser.add_argument('--policy_lr', default=0.0003, type=float,
                    help="Learning rate for agent (default: 0.0003).")
parser.add_argument('--alpha', default=0.2, type=float,
                    help="Alpha parameter for agent (default: 0.2).")
parser.add_argument('--auto-ent', default=False, action='store_true',
                    help="Enable automatic entropy tuning during agent training.")
parser.add_argument('--policy_train_batch_size', default=128, type=int,
                    help='batch size for training policy (default: 128)')

parser.add_argument('--num_epoch', type=int, default=100,
                    help='Total number of epochs')
parser.add_argument('--epoch_length', type=int, default=5000,
                    help='env steps per epoch')
parser.add_argument('--latent-m', default=False, action='store_true',
                    help="Incorporate latent representation into model.")
parser.add_argument('--latent-p', default=False, action='store_true',
                    help="Incorporate latent representation into policy.")
parser.add_argument('--latent-dim', default=128, type=int,
                    help="Dimension of latent representation (default: 128).")
parser.add_argument('--not-cont', default=False, action='store_true',
                    help="Disable continuity in training.")

# MBPO-specific arguments
parser.add_argument('--use_decay', type=bool, default=True, metavar='E',
                    help='use L2 regularization of ensemble nets weights to avoid overfitting')
parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                    help='ensemble size B (default: 7)')
parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                    help='number of top-performing ensemble models when making rollouts (default: 5)')
parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                    help='hidden size for predictive model')

# M_eff = rollout_batch_size / model_train_freq = 100,000 / 250 = 400 imaginary trajs per env step (on average)
parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                    help='number of imaginary trajs each time triggering model rollouts')
parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                    help='env step frequency of training model and triggering a rollout batch')
parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                    help='ratio of env samples / model samples for each policy-training mini-batch')
parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                    help='rollout min epoch for linearly increasing rollout length')
parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                    help='rollout max epoch for linearly increasing rollout length')
parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                    help='rollout min length for linearly increasing rollout length')
parser.add_argument('--rollout_max_length', type=int, default=1, metavar='A',
                    help='rollout max length for linearly increasing rollout length')
parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                    help='minimum pool size')
parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                    help='Keep only the last model_retain_epochs epochs’ model data in model pool.')

parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                    help='env step frequency of training policy')
parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                    help='times to training policy per step G')
parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                    help='max training times per step')
parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                    help='exploration steps initially')

def main(args=None):
    if args is None:
        args = parser.parse_args()

    ############
    # Init env
    ############
    env_class = ENV_DICT[args.env]
    env_params = ENV_PARAMS.get(args.env, dict())
    env = env_class(**env_params)
    env_name = env.get_env_name()
    F_A = env.F_A
    F_S = env.F_S
    NOISE_TYPE = args.noise_type
    print(f'Env: {env_name}')
    print(f'Noise type: {NOISE_TYPE}')

    # For swimmmer, dt = 0.01 * 4 (frame_skip)
    # For hopper, dt = 0.002 * 4
    # For walker2d, dt = 0.002 * 4
    dt_value = env.dt
    dt = torch.tensor(dt_value).to(device)

    print(f'Model type: {args.type}')
    if args.type == 'mbpo':
        pass
    else:
        if args.latent_p and not args.latent_m:
            raise ValueError("`latent_p=True` requires `latent_m=True`.")
        if args.type == 'mf' and args.latent_m:
            raise ValueError("latent representation is not supported when `type='mf'`.")
        TRAIN_SDE = 'sde' in args.type
        TRAIN_ODE = 'mb' in args.type
        CUT_LEN = args.h if args.cut_len is None else args.cut_len
        TF = True
        KL_COEFF = .99
        LATENT_DIM = args.latent_dim

        print(f'K: {args.k}')
        print(f'H: {args.h}')
        print(f'Alpha: {args.alpha}')
        print(f'Cut len: {CUT_LEN}')
        print(f'Use latent model: {args.latent_m}')
        print(f'Use latent policy: {args.latent_p}')
        print('\n')

    #######################
    # Init wandb project
    #######################
    run_name = f'{args.env}-{args.type}'
    model_type = args.type
    if model_type == 'mbpo':
        pass
    else:
        if args.latent_m:
            run_name += '-latent-model-policy' if args.latent_p else '-latent-model'
        if args.latent_m:
            model_type += '-LatMod'
        if args.latent_p:
            model_type += '-LatPol'

    wandb.init(
        project=f'mpc-{env_name}',
        name=run_name,
        config={
            'type': model_type,
            'env': args.env,
            'env_class': env_class,
            'env_params': env_params,
        }
    )

    ###############
    # Init agent
    ###############
    agent = SAC(
        num_inputs=(F_S + LATENT_DIM) if args.latent_p else F_S,
        action_space=env.action_space,
        gamma=0.99,
        tau=0.005,
        alpha=args.alpha,
        policy="Gaussian",
        target_update_interval=1,
        automatic_entropy_tuning=args.auto_ent,
        cuda=True,
        hidden_size=args.policy_hidden_size,
        lr=args.policy_lr,
    )

    if model_type == 'mbpo':
        from sac.replay_memory import ReplayMemory as ReplayMemory_mbpo
        from mbpo.model import EnsembleDynamicsModel
        from mbpo.sample_env import EnvSampler
        from mbpo.utils_train import train_mbpo
        from mbpo.predict_env import PredictEnv

        wandb.config.update({
            'num_networks': args.num_networks,
            'num_elites': args.num_elites,
            'model_train_freq': args.model_train_freq,
            'rollout_batch_size': args.rollout_batch_size,
            'epoch_length': args.epoch_length,
            'num_epoch': args.num_epoch,
            'real_ratio': args.real_ratio,
        })

        # Init ensemble nets
        env_model = EnsembleDynamicsModel(
            args.num_networks, args.num_elites, F_S, F_A,
            hidden_size=args.pred_hidden_size, use_decay=args.use_decay
        )

        # Modeled env dynamics
        predict_env = PredictEnv(env_model, env_name)

        # Init pool for env
        env_pool = ReplayMemory_mbpo(args.replay_size)
        # Init pool for model
        rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
        model_steps_per_epoch = int(1 * rollouts_per_epoch)
        new_pool_size = args.model_retain_epochs * model_steps_per_epoch
        model_pool = ReplayMemory_mbpo(new_pool_size)

        # Sampler of env
        env_sampler = EnvSampler(env)

        train_mbpo(args, env_sampler, predict_env, agent, env_pool, model_pool)
    else:
        wandb.config.update({
            'noise_type': NOISE_TYPE,
            'cut-len': CUT_LEN,
            'init_train_data': args.init_data,
            'init_val_data': args.init_data_val,
            'k': args.k,
            'h': args.h,
            'e': args.e,
            'agent_bs': args.policy_train_batch_size,
            'agent_lr': args.policy_lr,
            'alpha': args.alpha,
            'auto_ent': args.auto_ent,
            'latent_model': args.latent_m,
            'latent_policy': args.latent_p,
            'latent_dim': args.latent_dim,
            'not_cont': args.not_cont
        })

        ####################
        # Init ODE model
        ####################
        ENC_OUTPUT = LATENT_DIM
        ENCODER_TO_HIDDEN = 25
        if TRAIN_ODE:
            if args.latent_m:
                encoder = EncoderRNN(in_size=F_S + F_A, hidden_size=ENC_OUTPUT, num_layers=1)
                ode_model = LatentODE(
                    encoder,
                    f_s=F_S,
                    f_a=F_A,
                    dt_value=dt_value,
                    enc_output=ENC_OUTPUT,
                    latent_dim=LATENT_DIM,
                    encoder_to_hidden=ENCODER_TO_HIDDEN,
                    decoder_mlp_size=128,
                )
            else:
                ode_model = ODEBlock(
                    ODEFunc(
                        MLP(in_size=F_S + F_A, out_size=F_S, mlp_size=100, num_layers=3)
                    ),
                    dt=dt,
                    method="euler"
                )
            ode_model.to(device)

        ################################
        # Init model and agent buffers
        ################################
        # Trajs (states, actions, rnd) for model training
        memory_traj_train = ReplayMemory(1e5, TrajectoryPolicy)
        memory_traj_val = ReplayMemory(1e5, TrajectoryPolicy)
        # Transitions for agent training
        memory_trans = ReplayMemory(
            1e6,
            Transition if not args.latent_p else TransitionLatent
        )
        init_data_collection = args.init_data
        init_data_collection_val = args.init_data_val

        ####################
        # Init mbrl
        ####################
        mbrl = MBRL(
            env_model=ode_model if not args.type == 'mf' else None,
            agent=agent,
            memory_traj_train=memory_traj_train,
            memory_traj_val=memory_traj_val,
            memory_trans=memory_trans,
            env=env,
            f_s=F_S,
            f_a=F_A,
            std_act=False,
            latent_m=args.latent_m,
            latent_p=args.latent_p,
            TF=TF,
            KL_COEFF=KL_COEFF,
            latent_dim=LATENT_DIM,
        )

        ###########################################
        # Fill model buffers with preliminary data
        ###########################################
        # Pre-collect trajs using random policy and
        # save them in memory_traj for model training
        mbrl.curr_env_steps = 0
        while mbrl.curr_env_steps < init_data_collection:
            curr_env_steps = min(
                init_data_collection, init_data_collection - mbrl.curr_env_steps
            )
            mbrl.mpc_planning(
                updates=0,
                max_env_steps=curr_env_steps,
                cut_len=CUT_LEN,
                h=args.h,
                combine_mf=False,
                rand=True,
                val_ratio=0,
                rms=True,
                save_trans='mb' not in args.type,
            )
        # Num of trajs (init_data_collection // CUT_LEN)
        print('Init traj buffer size {} for model train'.format(mbrl.memory_traj_train.position))

        mbrl.curr_env_steps = 0
        while mbrl.curr_env_steps < init_data_collection_val:
            curr_env_steps = min(
                init_data_collection_val, init_data_collection_val - mbrl.curr_env_steps
            )
            mbrl.mpc_planning(
                updates=0,
                max_env_steps=curr_env_steps,
                cut_len=CUT_LEN,
                h=args.h,
                combine_mf=False,
                rand=True,
                val_ratio=1,
                rms=True,
                save_trans='mb' not in args.type,
            )
        # Num of trajs (init_data_collection_val // CUT_LEN)
        print('Init traj buffer size {} for model val\n'.format(mbrl.memory_traj_val.position))
        mbrl.calc_mean_std()
        wandb.config.update({
            'init_train_buffer_size': mbrl.memory_traj_train.position,
            'init_val_buffer_size': mbrl.memory_traj_val.position
        })

        ########################
        # Init ODE/SDE model
        ########################
        if TRAIN_ODE:
            LR = 1e-3
            WD = 1e-3
            optimizer = torch.optim.Adam(mbrl.env_model.parameters(), lr=LR, weight_decay=WD)

        if TRAIN_SDE:
            G_IN_DIM = (F_S + F_A + LATENT_DIM) if args.latent_m else (F_S + F_A)
            G_OUT_DIM = LATENT_DIM if args.latent_m else F_S
            min_values, max_values = mbrl.get_min_max()
            model_g = SDEBlockLatent(
                SDEFuncLatent(
                    f_model=mbrl.env_model,
                    g_out_dim=G_OUT_DIM,
                    g_model=nn.Sequential(
                        nn.Linear(G_IN_DIM, 100),
                        nn.Tanh(),
                        nn.Linear(100, 100),
                        nn.Tanh(),
                        nn.Linear(100, G_OUT_DIM)
                    ),
                    # g_proj_model=nn.Linear(G_OUT_DIM, G_OUT_DIM),
                    use_latent=args.latent_m,
                ),
                # latent_2_state_proj=nn.Sequential(
                #     nn.Linear(LATENT_DIM, 25),
                #     nn.LeakyReLU(),
                #     nn.Linear(25, F_S)
                # ),
                latent_2_state_proj=nn.Linear(LATENT_DIM, F_S),
                dt=dt,
                min_values=min_values,
                max_values=max_values,
                use_clip=True,
                noise_type=NOISE_TYPE,
            )
            model_d = CollapseDiscWLayerNorm(
                f_s=F_S, f_a=F_A, traj_len=args.h, mlp_size=100, use_action=False
            )

            G_LR = 8e-4
            D_LR = 4e-5
            mbrl.model_g = model_g.to(device)
            # Note mbrl.model_g.func.f_model is an ODEBlock or LatentODE
            mbrl.model_g.func.f_model.requires_grad_(False)
            mbrl.model_d = model_d.to(device)
            optimizer_g = torch.optim.Adam(model_g.parameters(), lr=G_LR, betas=(0, 0.9))
            optimizer_d = torch.optim.Adam(model_d.parameters(), lr=D_LR, betas=(0, 0.9))
            mbrl.set_visualization_datataset(rms=True)

        #######################################
        # Iterate model and agent over E epochs
        #######################################
        ts = torch.arange(args.h, device=device) * dt
        ts_ = ts.numpy(force=True)
        epoch_num = args.num_epoch  # Number of epochs (E)
        episode = 0                 # Episode counter over E epochs
        total_env_steps = 0         # Env step counter over E epochs
        total_ode_epochs = 0        # ODE-training epoch counter over E epochs
        total_sde_epochs = 0        # SDE-training epoch counter over E epochs
        SAVE_MODEL = True
        for iteration_idx in range(epoch_num):
            ################################
            # Update ODE with early stopping
            ################################
            if TRAIN_ODE:
                early_stopper = EarlyStopper(
                    patience=max(15-iteration_idx, 3), min_delta=0
                )

                total_ode_epochs += mbrl.train_ode_loop(
                    optimizer=optimizer,
                    ts=ts,
                    total_ode_epochs=total_ode_epochs,
                    n_epochs=500 if not args.not_cont else (500 if iteration_idx == 0 else 0),
                    batch_size=args.model_batch_size,
                    early_stopper=early_stopper,
                )

            #####################
            # Update SDE
            #####################
            if TRAIN_SDE:
                if iteration_idx == 0:
                    sde_epochs = 5000 if not args.not_cont else 10000
                else:
                    sde_epochs = 500 if not args.not_cont else 0

                total_sde_epochs += mbrl.train_sde_loop(
                    optimizer_d=optimizer_d,
                    optimizer_g=optimizer_g,
                    ts=ts,
                    ts_=ts_,
                    total_sde_epochs=total_sde_epochs,
                    n_epochs=sde_epochs,
                    batch_size=args.model_batch_size,
                )

            ##################################################
            # Update agent buffer and agent over M env steps
            ##################################################
            # Number of env steps per epoch (M)
            env_steps_global = args.epoch_length
            # Fixed max episodic steps
            max_steps = 1000
            mbrl.curr_env_steps = 0
            if 'mb' in args.type:
                while mbrl.curr_env_steps < env_steps_global:
                    actual_steps = min(
                        max_steps, env_steps_global - mbrl.curr_env_steps
                    )
                    rewards, env_steps = mbrl.mpc_planning(
                        updates=0,
                        max_env_steps=actual_steps,
                        cut_len=CUT_LEN,
                        batch_size=args.policy_train_batch_size,
                        k=args.k,
                        h=args.h,
                        e=args.e,
                        combine_mf=True,
                        rand=False,
                        val_ratio=.1,
                        rms=True,
                        save_trans=True,
                        save_trajs=True,
                        mode='mb'
                    )
                    episode += 1
                    total_env_steps += env_steps

                    log = "Episode {} | episodic steps = {} | episodic reward = {:.6f} | total env steps = {}".format(
                            episode, env_steps, rewards, total_env_steps
                    )
                    print(log)

                    if wandb.run is not None:
                        wandb.log(
                            dict(
                                total_env_steps=total_env_steps,
                                episode=episode,
                                rewards=rewards,
                                agent_update_steps=mbrl.update_steps
                            )
                        )
            elif 'mf' == args.type:
                while mbrl.curr_env_steps < env_steps_global:
                    actual_steps = min(
                        max_steps, env_steps_global - mbrl.curr_env_steps
                    )
                    rewards, env_steps = mbrl.mpc_planning(
                        updates=0,
                        max_env_steps=actual_steps,
                        cut_len=CUT_LEN,
                        batch_size=args.policy_train_batch_size,
                        combine_mf=True,
                        rand=False,
                        val_ratio=.1,
                        rms=True,
                        save_trans=True,
                        save_trajs=False,
                        mode='mf'
                    )
                    episode += 1
                    total_env_steps += env_steps

                    log = "Episode {} | episodic steps = {} | episodic reward = {:.6f} | total env steps = {}".format(
                        episode, env_steps, rewards, total_env_steps
                    )
                    print(log)

                    if wandb.run is not None:
                        wandb.log(
                            dict(
                                total_env_steps=total_env_steps,
                                episode=episode,
                                rewards=rewards,
                                agent_update_steps=mbrl.update_steps
                            )
                        )
            else:
                raise NotImplementedError(f'model must be either mf or mb but is now: {args.type}')

            ################################
            # End-of-epoch agent evaluation
            ################################
            eval_rewards = mbrl.eval_agent(n_episodes=25, rms=True).mean()
            if wandb.run is not None:
                wandb.log(
                    dict(
                        total_env_steps=total_env_steps,
                        episode=episode,
                        eval_rewards=eval_rewards,
                        agent_update_steps=mbrl.update_steps
                    )
                )
            print('Epoch(E) {} | Eval Rewards: {:.6f}'.format(iteration_idx, eval_rewards.mean()))

        if SAVE_MODEL and TRAIN_SDE :
            mbrl.save_model(
                env_name=env_name,
                run_id=wandb.run.id if wandb.run is not None else uuid4(),
            )

    wandb.finish()


if __name__ == "__main__":
    main()
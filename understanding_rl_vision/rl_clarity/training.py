import time
import os
import tempfile
import numpy as np
from mpi4py import MPI
import gym
from baselines.common.vec_env import (
    VecEnv,
    VecEnvWrapper,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from lucid.scratch.rl_util import save_joblib

PROCGEN_ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

PROCGEN_KWARG_KEYS = [
    "num_levels",
    "start_level",
    "fixed_difficulty",
    "use_easy_jump",
    "paint_vel_info",
    "use_generated_assets",
    "use_monochrome_assets",
    "restrict_themes",
    "use_backgrounds",
    "plain_assets",
    "is_high_difficulty",
    "is_uniform_difficulty",
    "distribution_mode",
    "use_sequential_levels",
    "fix_background",
    "physics_mode",
    "debug_mode",
    "center_agent",
    "env_name",
    "game_type",
    "game_mechanics",
    "sample_game_mechanics",
    "render_human",
]

ATARI_ENV_IDS = [
    "AirRaid",
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Carnival",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "DoubleDunk",
    "ElevatorAction",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "JourneyEscape",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "Pooyan",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
]

ATARI_ENV_DICT = {envid.lower(): envid for envid in ATARI_ENV_IDS}


# This is Thompson Sampling - Not working
# class EpsilonGreedy(VecEnvWrapper):
#     """
#     Override actions with Thompson Sampling
#
#     Args:
#         action_space: The action space of the environment
#         alpha_prior: Prior parameter for the Beta distribution (binary action space)
#         beta_prior: Prior parameter for the Beta distribution (binary action space)
#         alpha_mult: Prior parameter for the Dirichlet distribution (multi-dimensional action space)
#     """
#
#     def __init__(self, venv: VecEnv, epsilon: float):
#         super().__init__(venv)
#         self.alpha_prior = 1.0
#         self.beta_prior = 1.0
#         # self.alpha_mult = 1.0
#         assert isinstance(self.action_space, gym.spaces.Discrete) or isinstance(
#             self.action_space, gym.spaces.MultiBinary
#         )  # Only supports Discrete and MultiBinary action spaces
#
#     def reset(self):
#         return self.venv.reset()
#
#     def step_async(self, actions):
#         if isinstance(self.action_space, gym.spaces.Discrete):
#             sampled_actions = self.sample_discrete_thompson(actions)
#         elif isinstance(self.action_space, gym.spaces.MultiBinary):
#             sampled_actions = self.sample_binary_thompson(actions)
#         else:
#             raise NotImplementedError("Thompson Sampling does not support this action space.")
#         self.venv.step_async(sampled_actions)
#
#     def step_wait(self):
#         return self.venv.step_wait()
#
#     def sample_discrete_thompson(self, actions):
#         sampled_actions = []
#         for action_values in actions:
#             print(action_values)
#             action_probabilities = np.random.dirichlet([1])  # Uniform Dirichlet as alpha_mult isn't applicable
#             sampled_action = np.argmax(action_probabilities)
#             sampled_actions.append(sampled_action)
#         return np.array(sampled_actions)
#
#     def sample_binary_thompson(self, actions):
#         sampled_actions = []
#         for action_values in actions:
#             success_probabilities = (action_values + self.alpha_prior) / (
#                 action_values + self.alpha_prior + self.beta_prior
#             )
#             sampled_action = np.random.binomial(1, success_probabilities)
#             sampled_actions.append(sampled_action)
#         return np.array(sampled_actions)

# This is ThompsonSampling - This words but value function remains constant
# class EpsilonGreedy(VecEnvWrapper):
#     """
#     Override actions using Thompson Sampling.
#
#     Args:
#         venv: The vectorized environment to wrap.
#     """
#
#     def __init__(self, venv: VecEnv, epsilon: float):
#         super().__init__(venv)
#         assert isinstance(self.action_space, gym.spaces.Discrete), "Thompson Sampling only supports Discrete action spaces."
#
#     def reset(self):
#         return self.venv.reset()
#
#     def step_async(self, actions):
#         # Generate samples from posterior distribution for each arm
#         posterior_samples = np.random.beta(1, 1, size=(self.num_envs, self.action_space.n))
#
#         # Select actions with maximum sample
#         new_actions = np.argmax(posterior_samples, axis=1)
#
#         self.venv.step_async(new_actions)
#
#     def step_wait(self):
#         return self.venv.step_wait()

# This is ThompsonSampling - Not working - ValueError: Array passed to NMF (input H) is full of zeros.
# class EpsilonGreedy(VecEnvWrapper):
#     def __init__(self, venv: VecEnv, epsilon):
#         super().__init__(venv)
#         assert isinstance(self.action_space, gym.spaces.Discrete), "Thompson Sampling only supports Discrete action spaces."
#         init_alpha = 1.0
#         init_beta = 1.0
#         # Initialize alpha and beta parameters for each action in each environment
#         self.alpha = np.full((self.num_envs, self.action_space.n), init_alpha)
#         self.beta = np.full((self.num_envs, self.action_space.n), init_beta)
#         self.last_actions = np.zeros((self.num_envs,), dtype=int)
#
#     def reset(self):
#         # Reset alpha and beta to initial values on reset
#         self.alpha.fill(1.0)
#         self.beta.fill(1.0)
#         return self.venv.reset()
#
#     def step_async(self, actions):
#         # Thompson Sampling to select action based on the current belief (Beta distribution)
#         posterior_samples = np.random.beta(self.alpha, self.beta, size=(self.num_envs, self.action_space.n))
#         new_actions = np.argmax(posterior_samples, axis=1)
#         self.last_actions = np.argmax(posterior_samples, axis=1)
#         self.venv.step_async(new_actions)
#
#     def step_wait(self):
#         observations, rewards, dones, infos = self.venv.step_wait()
#
#         # Update alpha and beta based on the received rewards
#         for i, action in enumerate(self.last_actions):
#             # Update alpha if reward received, else update beta
#             self.alpha[i, action] += rewards[i]
#             self.beta[i, action] += 1 - rewards[i]
#
#         return observations, rewards, dones, infos

##############################
# This is EpochGreedy
# class EpsilonGreedy(VecEnvWrapper):
#     """
#     Override with random actions with probability epsilon for a whole epoch
#
#     Args:
#         epsilon: the probability actions will be overridden with random actions
#         epoch_length: number of steps in an epoch
#     """
#
#     def __init__(self, venv: VecEnv, epsilon: float):
#         super().__init__(venv)
#         assert isinstance(self.action_space, gym.spaces.Discrete) or isinstance(
#             self.action_space, gym.spaces.MultiBinary
#         )
#         self.epsilon = epsilon
#         self.epoch_length = 16
#         self.step_count = 0
#         self.prev_mask = np.random.uniform(size=self.num_envs)
#         # self.current_epoch_actions = None
#
#     def reset(self):
#         # self.step_count = 0
#         # self.current_epoch_actions = None
#         return self.venv.reset()
#
#     def step_async(self, actions):
#         if self.step_count % self.epoch_length == 0:
#             # Exploring
#             self.step_count = 0
#             # self.current_epoch_actions = np.random.uniform(size=self.num_envs) < self.epsilon
#             self.epsilon=0.95 #1-epsilon
#             mask = np.random.uniform(size=self.num_envs) < self.epsilon
#             print("Exploring")
#             # self.prev_mask = mask
#         else:
#             # Exploting
#             self.epsilon=0.05
#             mask = np.random.uniform(size=self.num_envs) < self.epsilon
#             print("Exploiting")
#             # mask = self.prev_mask
#         new_actions = np.array(
#             [
#                 self.action_space.sample() if mask[i] else actions[i]
#                 for i in range(self.num_envs)
#             ]
#         )
#         self.step_count += 1
#         print("\n\n************* STEP COUNT: ", self.step_count)
#         self.venv.step_async(new_actions)
#         # print("New actions: ", new_actions)
#
#     def step_wait(self):
#         # print(" Step wait: ",self.venv.step_wait())
#         return self.venv.step_wait()
##############################

# ORIGINAL!!!!
class EpsilonGreedy(VecEnvWrapper):
    """
    Overide with random actions with probability epsilon

    Args:
        epsilon: the probability actions will be overridden with random actions
    """

    def __init__(self, venv: VecEnv, epsilon: float):
        super().__init__(venv)
        print("\n\n\n******************Epsilon: ", epsilon)
        assert isinstance(self.action_space, gym.spaces.Discrete) or isinstance(
            self.action_space, gym.spaces.MultiBinary
        )
        self.epsilon = epsilon

    def reset(self):
        print("\n\n********** I AM IN reset\n\n")
        return self.venv.reset()

    def step_async(self, actions):
        # print("\n\n********** I AM IN step_async\n\n")
        mask = np.random.uniform(size=self.num_envs) < self.epsilon
        new_actions = np.array(
            [
                self.action_space.sample() if mask[i] else actions[i]
                for i in range(self.num_envs)
            ]
        )
        self.venv.step_async(new_actions)
        # print("New actions: ", new_actions)

    def step_wait(self):
        # print(" Step wait: ",self.venv.step_wait())
        return self.venv.step_wait()


class VecRewardScale(VecEnvWrapper):
    """
    Add `task_id` to the corresponding info dict of each environment
    in the provided VecEnv

    Args:
        venv: A set of environments
        task_ids: A list of task_ids corresponding to each environment in `venv`
    """

    def __init__(self, venv: VecEnv, scale: float):
        super().__init__(venv)
        self._scale = scale

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        rews = rews * self._scale
        return obs, rews, dones, infos


# our internal version of CoinRun old ended up with 2 additional actions, so
# the pre-trained models require this wrapper.
class VecExtraActions(VecEnvWrapper):
    def __init__(self, venv, *, extra_actions, default_action):
        assert isinstance(venv.action_space, gym.spaces.Discrete)
        super().__init__(
            venv, action_space=gym.spaces.Discrete(venv.action_space.n + extra_actions)
        )
        self.default_action = default_action

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        actions = actions.copy()
        for i in range(len(actions)):
            if actions[i] >= self.venv.action_space.n:
                actions[i] = self.default_action
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()


# hack to fix a bug caused by observations being modified in-place
class VecShallowCopy(VecEnvWrapper):
    def step_async(self, actions):
        actions = actions.copy()
        self.venv.step_async(actions)

    def reset(self):
        obs = self.venv.reset()
        return obs.copy()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return obs.copy(), rews.copy(), dones.copy(), infos.copy()


coinrun_initialized = False


def create_env(
    num_envs,
    *,
    env_kind="procgen",
    epsilon_greedy=0.05,             ### Changed
    reward_scale=1.0,
    frame_stack=1,
    use_sticky_actions=0,
    coinrun_old_extra_actions=0,
    **kwargs,
):
    if env_kind == "procgen":
        env_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        env_name = env_kwargs.pop("env_name")

        if env_name == "coinrun_old":
            import coinrun
            from coinrun.config import Config

            Config.initialize_args(use_cmd_line_args=False, **env_kwargs)
            global coinrun_initialized
            if not coinrun_initialized:
                coinrun.init_args_and_threads()
                coinrun_initialized = True
            venv = coinrun.make("standard", num_envs)
            if coinrun_old_extra_actions > 0:
                venv = VecExtraActions(
                    venv, extra_actions=coinrun_old_extra_actions, default_action=0
                )

        else:
            from procgen import ProcgenGym3Env
            import gym3

            env_kwargs = {
                k: v for k, v in env_kwargs.items() if k in PROCGEN_KWARG_KEYS
            }
            env = ProcgenGym3Env(num_envs, env_name=env_name, **env_kwargs)
            env = gym3.ExtractDictObWrapper(env, "rgb")
            venv = gym3.ToBaselinesVecEnv(env)

    elif env_kind == "atari":
        game_version = "v0" if use_sticky_actions == 1 else "v4"

        def make_atari_env(lower_env_id, num_env):
            env_id = ATARI_ENV_DICT[lower_env_id] + f"NoFrameskip-{game_version}"

            def make_atari_env_fn():
                env = make_atari(env_id)
                env = wrap_deepmind(env, frame_stack=False, clip_rewards=False)

                return env

            return SubprocVecEnv([make_atari_env_fn for i in range(num_env)])

        lower_env_id = kwargs["env_id"]

        venv = make_atari_env(lower_env_id, num_envs)

    else:
        raise ValueError(f"Unsupported env_kind: {env_kind}")

    if frame_stack > 1:
        venv = VecFrameStack(venv=venv, nstack=frame_stack)

    if reward_scale != 1:
        venv = VecRewardScale(venv, reward_scale)

    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

    if epsilon_greedy > 0:
        venv = EpsilonGreedy(venv, epsilon_greedy)

    venv = VecShallowCopy(venv)

    return venv


def get_arch(
    *,
    library="baselines",
    cnn="clear",
    use_lstm=0,
    stack_channels="16_32_32",
    emb_size=256,
    **kwargs,
):
    stack_channels = [int(x) for x in stack_channels.split("_")]

    if library == "baselines":
        if cnn == "impala":
            from baselines.common.models import build_impala_cnn

            conv_fn = lambda x: build_impala_cnn(
                x, depths=stack_channels, emb_size=emb_size
            )
        elif cnn == "nature":
            from baselines.common.models import nature_cnn

            conv_fn = nature_cnn
        elif cnn == "clear":
            from lucid.scratch.rl_util.arch import clear_cnn

            conv_fn = clear_cnn
        else:
            raise ValueError(f"Unsupported cnn: {cnn}")

        if use_lstm:
            from baselines.common.models import cnn_lstm

            arch = cnn_lstm(nlstm=256, conv_fn=conv_fn)
        else:
            arch = conv_fn

    else:
        raise ValueError(f"Unsupported library: {library}")

    return arch


def create_tf_session():
    """
    Create a TensorFlow session
    """
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def get_tf_params(scope):
    """
    Get a dictionary of parameters from TensorFlow for the specified scope
    """
    import tensorflow as tf
    from baselines.common.tf_util import get_session

    sess = get_session()
    allvars = tf.trainable_variables(scope)
    nonopt_vars = [
        v
        for v in allvars
        if all(veto not in v.name for veto in ["optimizer", "kbuf", "vbuf"])
    ]
    name2var = {v.name: v for v in nonopt_vars}
    return sess.run(name2var)


def save_data(*, save_dir, args_dict, params, step=None, extra={}):
    """
    Save the global config object as well as the current model params to a local file
    """
    data_dict = dict(args=args_dict, params=params, extra=extra, time=time.time())

    step_str = "" if step is None else f"-{step}"
    save_path = os.path.join(save_dir, f"checkpoint{step_str}.jd")

    if "://" not in save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save_joblib(data_dict, save_path)

    return save_path


class VecClipReward(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        """Bin reward to {+1, 0, -1} by its sign."""
        obs, rews, dones, infos = self.venv.step_wait()
        return obs, np.sign(rews), dones, infos


def train(comm=None, *, save_dir=None, **kwargs):
    """
    Train a model using Baselines' PPO2, and to save a checkpoint file in the
    required format.

    There is one required kwarg: either env_name (for env_kind="procgen") or
    env_id (for env_kind="atari").

    Models for the paper were trained with 16 parallel MPI workers.

    Note: this code has been well-tested.
    """
    kwargs.setdefault("env_kind", "procgen")
    kwargs.setdefault("num_envs", 64)
    kwargs.setdefault("learning_rate", 5e-4)
    kwargs.setdefault("entropy_coeff", 0.01)
    kwargs.setdefault("gamma", 0.999)
    kwargs.setdefault("lambda", 0.95)
    kwargs.setdefault("num_steps", 256)
    kwargs.setdefault("num_minibatches", 8)
    kwargs.setdefault("library", "baselines")
    kwargs.setdefault("save_all", False)
    kwargs.setdefault("ppo_epochs", 3)
    kwargs.setdefault("clip_range", 0.2)
    kwargs.setdefault("timesteps_per_proc", 1_000_000_000)
    kwargs.setdefault("cnn", "clear")
    kwargs.setdefault("use_lstm", 0)
    kwargs.setdefault("stack_channels", "16_32_32")
    kwargs.setdefault("emb_size", 256)
    kwargs.setdefault("epsilon_greedy", 0.05)             ########### CHanged
    kwargs.setdefault("reward_scale", 1.0)
    kwargs.setdefault("frame_stack", 1)
    kwargs.setdefault("use_sticky_actions", 0)
    kwargs.setdefault("clip_vf", 1)
    kwargs.setdefault("reward_processing", "none")
    kwargs.setdefault("save_interval", 10)

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    setup_mpi_gpus()

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="rl_clarity_train_")

    create_env_kwargs = kwargs.copy()
    num_envs = create_env_kwargs.pop("num_envs")
    venv = create_env(num_envs, **create_env_kwargs)

    library = kwargs["library"]
    if library == "baselines":
        reward_processing = kwargs["reward_processing"]
        if reward_processing == "none":
            pass
        elif reward_processing == "clip":
            venv = VecClipReward(venv=venv)
        elif reward_processing == "normalize":
            venv = VecNormalize(venv=venv, ob=False, per_env=False)
        else:
            raise ValueError(f"Unsupported reward processing: {reward_processing}")

        scope = "ppo2_model"

        def update_fn(update, params=None):
            if rank == 0:
                save_interval = kwargs["save_interval"]
                if save_interval > 0 and update % save_interval == 0:
                    print("Saving...")
                    params = get_tf_params(scope)
                    save_path = save_data(
                        save_dir=save_dir,
                        args_dict=kwargs,
                        params=params,
                        step=(update if kwargs["save_all"] else None),
                    )
                    print(f"Saved to: {save_path}")

        sess = create_tf_session()
        sess.__enter__()

        if kwargs["use_lstm"]:
            raise ValueError("Recurrent networks not yet supported.")
        arch = get_arch(**kwargs)

        from baselines.ppo2 import ppo2

        ppo2.learn(
            env=venv,
            network=arch,
            total_timesteps=kwargs["timesteps_per_proc"],
            save_interval=0,
            nsteps=kwargs["num_steps"],
            nminibatches=kwargs["num_minibatches"],
            lam=kwargs["lambda"],
            gamma=kwargs["gamma"],
            noptepochs=kwargs["ppo_epochs"],
            log_interval=1,
            ent_coef=kwargs["entropy_coeff"],
            mpi_rank_weight=1.0,
            clip_vf=bool(kwargs["clip_vf"]),
            comm=comm,
            lr=kwargs["learning_rate"],
            cliprange=kwargs["clip_range"],
            update_fn=update_fn,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    else:
        raise ValueError(f"Unsupported library: {library}")

    return save_dir

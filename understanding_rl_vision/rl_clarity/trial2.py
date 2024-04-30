from understanding_rl_vision import rl_clarity
import os

# The following program runs the training and inference of Atari Enduro

path="/home/rahul/rl_projects/pacman/Pacman_RL_with_feedback/src/understanding-rl-vision/understanding_rl_vision/rl_clarity/checkpoints/coinrun.jd"
odir="/home/rahul/rl_projects/pacman/Pacman_RL_with_feedback/src/understanding-rl-vision/understanding_rl_vision/rl_clarity/odir"

# Training and saving checkpoints in new file
# rl_clarity.train(env_name='coinrun', save_dir=path,num_envs=1,num_steps=32)

#rl_clarity.run(path, output_dir=odir)
# checkpoint_path = '/tmp/rl_clarity_example_vakoabg5/training/checkpoint.jd'
# checkpoint_path = "/tmp/rl_clarity_example_6jqoe1ml/training/checkpoint.jd" # ORIGINAL Epsilon Greedy 16 and 1
# checkpoint_path = "/tmp/rl_clarity_example_4ao3b635/training/checkpoint.jd" # ThompsonSampling - one with alpha beta update - 16 and 1
# checkpoint_path = "/tmp/rl_clarity_example_jb8vbg2k/training/checkpoint.jd" # Epoch Greedy with 16 as epoch length
# checkpoint_path = "/tmp/rl_clarity_example__bf6rpxy/training/checkpoint.jd" # EpochGreedy with 3 as epoch length and reset set
checkpoint_path = "/tmp/rl_clarity_example_pf9nt1qu/training/checkpoint.jd" # EpochGreedy with 3 as epoch and Rahul changes to mask
#checkpoint_path=path
interface_dir = odir
print("Generating interface...")
# generate a small interface, to demonstrate
print(checkpoint_path)
print(interface_dir)
rl_clarity.run(
    checkpoint_path,
    output_dir=interface_dir,
    trajectories_kwargs={"num_envs": 1, "num_steps": 16},   ######## Steps changed
    observations_kwargs={"num_envs": 1, "num_obs": 16, "obs_every": 4},
    layer_kwargs={"name_contains_one_of": ["2b"]},
)
interface_path = os.path.join(interface_dir, "interface.html")
interface_url = ("" if "://" in interface_path else "file://") + interface_path
print(f"Interface URL: {interface_url}")

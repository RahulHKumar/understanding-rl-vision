from understanding_rl_vision import rl_clarity

# The following program runs the inference of CoinRun using checkpoints available

path="/home/rahul/rl_projects/pacman/Pacman_RL_with_feedback/src/understanding-rl-vision/understanding_rl_vision/rl_clarity/checkpoints/coinrun.jd"
odir="/home/rahul/rl_projects/pacman/Pacman_RL_with_feedback/src/understanding-rl-vision/understanding_rl_vision/rl_clarity/odir"
rl_clarity.run(path, output_dir=odir)

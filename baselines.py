import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from gym_env import *
from data import *
import torch
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
import pathlib, os
from datetime import datetime
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path = "C:\\Users\emire\\vs_repositories\\trade_bot\\runs\\" + datetime.today().strftime("20%y_%m_%d__%H_%M")
pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True) 
writer = SummaryWriter(log_dir=path)
env = Environment(writer=writer)
eval_env = Environment(is_eval=True, writer=writer)
eval_callback = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=50000,
                             deterministic=True, render=False)

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
model_name = "DDPG_initial"
if pathlib.Path("C:\\Users\emire\\vs_repositories\\trade_bot\\models\\"+model_name).is_file():
    model = PPO.load("C:\\Users\emire\\vs_repositories\\trade_bot\\models\\"+model_name, env=env)
    print(f"==========load model {model_name}===============")
else:
    model = A2C('MlpPolicy', env, verbose=1, learning_rate=0.0003)
    print("=============creating new model===============")

model.learn(total_timesteps=500000, callback=eval_callback)

model.save(path="C:\\Users\emire\\vs_repositories\\trade_bot\\models\\"+model_name)

plot_results([log_dir], 5000, results_plotter.X_TIMESTEPS, "Bot")
plt.show()




"""
Source: https://huggingface.co/learn/deep-rl-course/unit1/hands-on
"""

# Virtual display
from pyvirtualdisplay import Display
import gymnasium

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")


# Load the saved model
model = PPO.load("pph-lunarland-v3", env=env)


# Evaluate the loaded model
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
observation, info = vec_env.reset(seed=42)
for i in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = vec_env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = vec_env.reset()

env.close()

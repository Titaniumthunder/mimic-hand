import os
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from grasp_env import HandGraspEnv

class RewardCallback(BaseCallback):
    def __init__(self, save_path):
        super().__init__()
        self.episode_rewards = []
        self.save_path = save_path
        self.fig, self.ax = plt.subplots()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                print(f"Episode {len(self.episode_rewards):4d} | "
                      f"Reward: {ep_reward:8.2f} | "
                      f"Timestep: {self.num_timesteps:7d}")
                self._update_plot()
        return True

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("PPO Mimic Hand — Episode Reward")
        eps = list(range(1, len(self.episode_rewards) + 1))
        self.ax.plot(eps, self.episode_rewards, alpha=0.4, color="steelblue", label="Reward")
        if len(self.episode_rewards) >= 20:
            rolling = [
                sum(self.episode_rewards[max(0, i-19):i+1]) / min(20, i+1)
                for i in range(len(self.episode_rewards))
            ]
            self.ax.plot(eps, rolling, color="orange", linewidth=2, label="20-ep mean")
        self.ax.legend()
        self.fig.savefig(self.save_path, dpi=100)

def main():
    N_ENVS      = 128    #hand sim is heavier
    TOTAL_STEPS = 20_000_000
    MODEL_PATH  = "hand_ppo"
    PLOT_PATH   = "hand_reward_plot.png"

    os.makedirs("checkpoints", exist_ok=True)

    print(f"Creating {N_ENVS} parallel environments...")
    vec_env = make_vec_env(HandGraspEnv, n_envs=N_ENVS)

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=vec_env, device="mps")
    else:
        print("Starting fresh...")
        model = PPO(
            "MlpPolicy",       # not CnnPolicy — observations are numbers not pixels
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            device="mps",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path="./checkpoints/",
        name_prefix="hand_ppo",
    )
    reward_cb = RewardCallback(save_path=PLOT_PATH)

    print(f"Training for {TOTAL_STEPS:,} timesteps...\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_cb, reward_cb])

    model.save(MODEL_PATH)
    vec_env.close()
    print(f"\nDone! Model saved → {MODEL_PATH}.zip")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
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
        self.ax.set_title("SAC Mimic Hand — Episode Reward")
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
    TOTAL_STEPS = 1_000_000
    MODEL_PATH  = "hand_sac"
    PLOT_PATH   = "hand_reward_plot.png"

    os.makedirs("checkpoints", exist_ok=True)

    # SAC requires a single (non-vectorized) env
    env = HandGraspEnv()

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing model...")
        model = SAC.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("Starting fresh...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,   # replay buffer — key SAC advantage over PPO
            learning_starts=10_000,  # collect experience before first update
            batch_size=256,
            tau=0.005,               # soft target update
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",         # automatic entropy tuning
            device="cpu",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="hand_sac",
    )
    reward_cb = RewardCallback(save_path=PLOT_PATH)

    print(f"Training for {TOTAL_STEPS:,} timesteps...\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_cb, reward_cb])

    model.save(MODEL_PATH)
    env.close()
    print(f"\nDone! Model saved → {MODEL_PATH}.zip")

if __name__ == "__main__":
    main()

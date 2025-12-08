import sys

import torch

sys.path.append("./iql")
from iql.IQL import IQL
from jetrover_gym.manipulation_env import JetRoverManipulationEnv
import numpy as np

MODEL_PATH = "iql/results/models/actor_s500000.pth"

def main():
    # Hyperparameters
    expectile = 0.7
    temperature = 3.0
    tau = 0.005
    discount = 0.99
    max_timesteps = 100

    env = JetRoverManipulationEnv()
    state_dim = env.observation_space
    action_dim = env.action_space
    
    print(f"Model Path: {MODEL_PATH}")

    obs, _ = env.reset()

    state_dim = 7
    action_dim = 7

    policy = IQL(state_dim=state_dim,
                 action_dim=action_dim,
                 expectile=expectile,
                 discount=discount,
                 tau=tau,
                 temperature=temperature)

    policy.actor.load_state_dict(torch.load(MODEL_PATH))
    
    print("Initializing: Forcing Gripper Open...")
    
    # 7개의 action 값: [x, y, z, r, p, y, gripper]
    # 이동은 0.0, 그리퍼는 0.0 (Open)으로 설정
    # (환경 설정에 따라 0.0이 Open, 1.0이 Close인 경우가 많음)
    open_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    # 확실하게 열리도록 5~10 step 정도 실행
    for _ in range(10):
        env.step(open_action)

    for _ in range(max_timesteps):
        obs = obs['state']
        action = policy.select_action(obs)
        obs, _, _, _, _ = env.step(action)
        print("ACTION:", action)
    env.close()

if __name__ == "__main__":
    main()

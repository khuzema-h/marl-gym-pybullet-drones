
import os
import time
import argparse
import numpy as np
import gymnasium as gym
import torch
import wandb
import sys

# Ensure gym_pybullet_drones is importable and safe_control_gym is found
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'gym_pybullet_drones'))

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.mappo.mappo import MAPPO
from gym_pybullet_drones.mappo.config import MAPPO_CONFIG as mappo_config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_eval(checkpoint_path, num_episodes=10, num_drones=2, gui=True, use_wandb=True, hidden_dim=64, act_type_str='rpm'):
    
    # Map action type string to enum
    act_str_lower = act_type_str.lower()
    if act_str_lower == 'rpm':
        act_type = ActionType.RPM
    elif act_str_lower == 'pid':
        act_type = ActionType.PID
    elif act_str_lower == 'vel':
        act_type = ActionType.VEL
    elif act_str_lower == 'one_d_rpm':
        act_type = ActionType.ONE_D_RPM
    elif act_str_lower == 'one_d_pid':
        act_type = ActionType.ONE_D_PID
    else:
        print(f"[WARN] Unknown action type '{act_type_str}', defaulting to RPM")
        act_type = ActionType.RPM

    # Initialize WandB
    if use_wandb:
        run_name = f"eval_{int(time.time())}"
        wandb.init(
            project="gym-pybullet-drones-mappo-eval",
            name=run_name,
            config={
                "checkpoint": checkpoint_path,
                "num_episodes": num_episodes,
                "num_drones": num_drones,
                "gui": gui,
                "hidden_dim": hidden_dim,
                "act_type": str(act_type)
            }
        )

    # Create environment
    print(f"[INFO] Creating environment with {num_drones} drones, GUI={gui}, ActionType={act_type}")
    env = MultiHoverAviary(
        num_drones=num_drones,
        gui=gui,
        obs=ObservationType.KIN,
        act=act_type,
        record=False
    )

    # Factory for internal MAPPO env (needs matching config)
    def env_factory(**kwargs):
        return MultiHoverAviary(
            num_drones=num_drones,
            gui=False,
            obs=ObservationType.KIN,
            act=act_type
        )

    print(f"[INFO] Initializing MAPPO agent with hidden_dim={hidden_dim}...")
    
    # Update config with correct hidden dim
    config = mappo_config.copy()
    config['hidden_dim'] = hidden_dim

    mappo_controller = MAPPO(
        env_func=env_factory,
        training=False,
        checkpoint_path=checkpoint_path,
        **config
    )
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    try:
        mappo_controller.load(checkpoint_path)
    except RuntimeError as e:
        print(f"\n[ERROR] Failed to load checkpoint: {e}")
        print("\n[HINT] Common configuration mismatches:")
        print("  - If you see size mismatch (256, XX) vs (64, XX), try: --hidden_dim 256")
        print("  - If you see obs mismatch involving 27 vs 72:")
        print("      * This means checkpoint used 1D actions (12 state + 15 buffer = 27)")
        print("      * But current env uses 4D actions (12 state + 60 buffer = 72)")
        print("      * TRY RUNNING WITH: --act one_d_rpm")
        sys.exit(1)
    
    agent = mappo_controller.agent
    
    # Set to eval mode
    agent.eval()
    
    # Evaluation Loop
    print(f"[INFO] Starting evaluation for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        terminated = False
        truncated = False
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            with torch.no_grad():
                # Get action from agent
                # Ensure input is convertible to tensor
                actions = mappo_controller.agent.ac.act(obs)
            
            # Step env
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if gui:
                time.sleep(1/env.CTRL_FREQ)

        print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.4f} | Length: {steps}")
        if "termination_reasons" in info and info["termination_reasons"]:
            print(f"  [TERMINATION] Reasons: {info['termination_reasons']}")
        
        if use_wandb:
            wandb.log({
                "eval/episode_reward": total_reward,
                "eval/episode_length": steps,
                "eval/episode": episode + 1
            })

    env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAPPO evaluation script')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--num_drones', type=int, default=2, help='Number of drones')
    parser.add_argument('--gui', type=str2bool, default=True, help='Enable GUI')
    parser.add_argument('--use_wandb', type=str2bool, default=True, help='Enable WandB logging')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size (must match training)')
    parser.add_argument('--act', type=str, default='rpm', help='Action type (rpm, pid, vel, one_d_rpm, one_d_pid)')

    args = parser.parse_args()
    
    run_eval(
        checkpoint_path=args.checkpoint_path,
        num_episodes=args.num_episodes,
        num_drones=args.num_drones,
        gui=args.gui,
        use_wandb=args.use_wandb,
        hidden_dim=args.hidden_dim,
        act_type_str=args.act
    )

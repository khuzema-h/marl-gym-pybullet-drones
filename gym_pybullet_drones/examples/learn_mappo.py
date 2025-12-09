"""Script demonstrating the use of `gym_pybullet_drones` with custom modular PPO implementation."""

import os
import sys
import time
from datetime import datetime
import argparse

# Add the parent directory to Python path to find the ppo module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This goes to gym_pybullet_drones
sys.path.insert(0, parent_dir)

# Now import other modules
import gymnasium as gym
import numpy as np
import torch

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available, continuing without it...")

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Import from your modular PPO implementation
try:
    from mappo import MAPPO, MAPPO_CONFIG
    from mappo.agent import MAPPOAgent
    from mappo.buffer import MAPPOBuffer
    print("✓ Successfully imported MAPPO modules")
except ImportError as e:
    print(f"✗ Error importing MAPPO modules: {e}")
    print("Looking for MAPPO module in:")
    print(f"  Current directory: {current_dir}")
    print(f"  Parent directory: {parent_dir}")
    print("Available directories in parent:")
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}/")
        else:
            print(f"  - {item}")
    sys.exit(1)

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 3
DEFAULT_MA = False

# WandB configuration
DEFAULT_USE_WANDB = True and WANDB_AVAILABLE
DEFAULT_WANDB_PROJECT = "gym-pybullet-drones-mappo"
DEFAULT_WANDB_ENTITY = None

def create_env(multiagent=False, gui=False, record_video=False, num_drones=1, seed=None):
    """Create the appropriate environment."""
    if multiagent:
        return MultiHoverAviary(
            gui=gui,
            num_drones=num_drones,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video
        )
    else:
        return HoverAviary(
            gui=gui,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video
        )

def run(multiagent=DEFAULT_MA, 
        output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        plot=True, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True,
        use_wandb=DEFAULT_USE_WANDB,
        wandb_project=DEFAULT_WANDB_PROJECT,
        wandb_entity=DEFAULT_WANDB_ENTITY):

    # Initialize WandB
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("WandB requested but not available. Continuing without WandB...")
        use_wandb = False

    # Create output directory
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.join(output_folder, f'save-{timestamp}')
    if not os.path.exists(filename):
        os.makedirs(filename)

    # Create environment function for PPO that accepts seed parameter
    def env_func(seed=None, **kwargs):
        return create_env(multiagent=multiagent, gui=False, record_video=False, 
                         num_drones=DEFAULT_AGENTS if multiagent else 1, seed=seed)

    # Custom PPO configuration
    mappo_config = MAPPO_CONFIG.copy()
    mappo_config.update({
        'output_dir': filename,
        'checkpoint_path': os.path.join(filename, 'model_latest.pt'),
        'max_env_steps': int(1e7) if local else int(1e2),
        'eval_interval': 10000,
        'eval_batch_size': 5,
        'log_interval': 1,
        'save_interval': 50000,
        'tensorboard': use_wandb,
        # You can customize other hyperparameters here
        'hidden_dim': 64,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'rollout_steps': 2048,
        'rollout_batch_size': 1,  # Number of parallel environments
    })

    # Log configuration to WandB
    if use_wandb:
        wandb_run.config.update(mappo_config)

    # Initialize PPO controller
    print("Initializing PPO controller...")
    mappo_controller = MAPPO(
        env_func=env_func,
        training=True,
        **mappo_config
    )

    # Reset the environment
    mappo_controller.reset()

    # Training
    print("Starting training...")
    try:
        mappo_controller.learn()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

    # Save final model
    final_model_path = os.path.join(filename, 'final_model.pt')
    mappo_controller.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    ############################################################
    # Evaluation and Demonstration
    ############################################################

    # Load the best model for evaluation
    best_model_path = os.path.join(filename, 'model_best.pt')
    if os.path.exists(best_model_path):
        print("Loading best model for evaluation...")
        mappo_controller.load(best_model_path)
    else:
        print("Best model not found, using final model...")
        mappo_controller.load(final_model_path)

    # Create test environment
    print("Creating test environment...")
    test_env = create_env(
        multiagent=multiagent, 
        gui=gui, 
        record_video=record_video,
        num_drones=DEFAULT_AGENTS if multiagent else 1
    )

    # Setup logger
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=DEFAULT_AGENTS if multiagent else 1,
        output_folder=output_folder,
        colab=colab,
        use_wandb=use_wandb,
        wandb_run=wandb_run
    )

    # Run evaluation
    print("Running evaluation...")
    eval_results = mappo_controller.run(
        env=test_env, 
        n_episodes=5,
        render=False  # Don't render during batch evaluation
    )
    
    mean_reward = eval_results['ep_returns'].mean()
    std_reward = eval_results['ep_returns'].std()
    print(f"\nEvaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode lengths: {eval_results['ep_lengths'].mean():.2f} +/- {eval_results['ep_lengths'].std():.2f}")

    # Log evaluation results to WandB
    if use_wandb:
        wandb_run.log({
            "final_evaluation/mean_reward": mean_reward,
            "final_evaluation/std_reward": std_reward,
            "final_evaluation/mean_episode_length": eval_results['ep_lengths'].mean(),
        })

    # Demonstration run with rendering
    print("\nRunning demonstration...")
    obs, info = test_env.reset(seed=42)
    start = time.time()
    
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action = mappo_controller.select_action(obs, info)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        # Log data for visualization
        obs_flat = obs.squeeze()
        action_flat = action.squeeze()
        
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(
                    drone=0,
                    timestamp=i / test_env.CTRL_FREQ,
                    state=np.hstack([
                        obs_flat[0:3],        # position
                        np.zeros(4),          # orientation (quat) - not used
                        obs_flat[3:15],       # velocity, angular velocity, etc.
                        action_flat           # action
                    ]),
                    control=np.zeros(12)      # control signals
                )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(
                        drone=d,
                        timestamp=i / test_env.CTRL_FREQ,
                        state=np.hstack([
                            obs_flat[d][0:3],    # position
                            np.zeros(4),         # orientation
                            obs_flat[d][3:15],   # velocity, angular velocity
                            action_flat[d]       # action
                        ]),
                        control=np.zeros(12)     # control signals
                    )
        
        # Render and sync
        test_env.render()
        print(f"Step: {i}, Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")
        
        # Log to WandB periodically
        if use_wandb and i % 10 == 0:
            wandb_run.log({
                "demonstration/reward": reward,
                "demonstration/step": i
            })
        
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            print("Episode finished!")
            obs, info = test_env.reset(seed=42)
    
    test_env.close()

    # Generate plots
    if plot and DEFAULT_OBS == ObservationType.KIN:
        print("Generating plots...")
        logger.plot()

    # Finish WandB run
    if use_wandb:
        wandb_run.finish()

    print(f"\nTraining and evaluation completed!")
    print(f"Results saved to: {filename}")

def test_mappo_components():
    """Test function to verify MAPPO components are working correctly."""
    print("Testing MAPPO components...")
    
    # Test buffer
    try:
        import gymnasium as gym
        test_env = gym.make('Pendulum-v1')
        buffer = MAPPOBuffer(
            test_env.observation_space,
            test_env.action_space,
            max_length=100,
            batch_size=1
        )
        print("✓  MAPPOBuffer initialized successfully")
        
        # Test agent
        agent = MAPPOAgent(
            test_env.observation_space,
            test_env.action_space,
            hidden_dim=64
        )
        print("✓ MAPPOAgent initialized successfully")
        
        test_env.close()
        print("✓ All MAPPO components working correctly!")
        
    except Exception as e:
        print(f"✗ Error testing MAPPO components: {e}")
        raise

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='PPO training script for drone control')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use multi-agent environment (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--use_wandb',          default=DEFAULT_USE_WANDB,     type=str2bool,      help='Whether to use Weights & Biases for logging (default: True)', metavar='')
    parser.add_argument('--wandb_project',      default=DEFAULT_WANDB_PROJECT, type=str,           help='WandB project name (default: "gym-pybullet-drones-ppo")', metavar='')
    parser.add_argument('--wandb_entity',       default=DEFAULT_WANDB_ENTITY,  type=str,           help='WandB entity (team name) (default: None)', metavar='')
    parser.add_argument('--test_components',    default=False,                 type=str2bool,      help='Test PPO components before training (default: False)', metavar='')
    
    ARGS = parser.parse_args()

    # Test components if requested
    if ARGS.test_components:
        test_mappo_components()
        print("Component testing completed. Starting main training...")
    
    # Remove test_components from args before passing to run()
    args_dict = vars(ARGS)
    args_dict.pop('test_components', None)
    
    # Run main training
    run(**args_dict)
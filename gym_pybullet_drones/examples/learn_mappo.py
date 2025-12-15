"""Script demonstrating the use of `gym_pybullet_drones` with custom modular MAPPO implementation."""

import os
import sys
import time
from datetime import datetime
import argparse

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
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

# SIMPLE DIRECT IMPORT
try:
    # Import the mappo package - this should work now
    import mappo
    # Access the classes we need
    MAPPO = mappo.MAPPO
    MAPPO_CONFIG = mappo.MAPPO_CONFIG
    print("✓ Successfully imported MAPPO modules")
    
    # Debug: print what we imported
    print(f"  MAPPO: {MAPPO}")
    print(f"  MAPPO_CONFIG keys: {list(MAPPO_CONFIG.keys())[:10]}...")
    
except ImportError as e:
    print(f"✗ Error importing MAPPO modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Default parameters

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_pid')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 5  # Start with 2 agents for easier training
DEFAULT_MA = True  # Default to multi-agent for MAPPO

# WandB configuration
DEFAULT_USE_WANDB = True and WANDB_AVAILABLE
DEFAULT_WANDB_PROJECT = "gym-pybullet-drones-mappo"
DEFAULT_WANDB_ENTITY = None

def create_env(multiagent=False, gui=False, record_video=False, num_drones=1, seed=None):
    """Create the appropriate environment."""
    if multiagent:
        env = MultiHoverAviary(
            gui=gui,
            num_drones=num_drones,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video
        )
        # Add method to get global state for centralized critic
        if not hasattr(env, 'get_global_state'):
            env.get_global_state = lambda: env._getDroneStateVector(0)  # Simple placeholder
        if not hasattr(env, 'get_global_state_dim'):
            env.get_global_state_dim = lambda: 20 * num_drones  # Approximate global state dimension
        return env
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
        wandb_entity=DEFAULT_WANDB_ENTITY,
        rollout_batch_size=128,
        num_workers=4,
        use_gpu=True,
        num_drones=None):

    # Set number of drones/agents
    global DEFAULT_AGENTS
    if num_drones is None:
        num_drones = DEFAULT_AGENTS
    else:
        # Update DEFAULT_AGENTS for this run
        DEFAULT_AGENTS = num_drones
    
    print(f"✓ Training with {num_drones} agents")
    
    # Initialize WandB
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            config={
                "multiagent": multiagent,
                "num_agents": num_drones if multiagent else 1,
                "obs_type": str(DEFAULT_OBS),
                "act_type": str(DEFAULT_ACT)
            }
        )
        print(f"✓ WandB initialized: {wandb_run.name}")
    elif use_wandb and not WANDB_AVAILABLE:
        print("WandB requested but not available. Continuing without WandB...")
        use_wandb = False

    # Create output directory
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.join(output_folder, f'mappo-save-{timestamp}')
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    print(f"✓ Output directory created: {filename}")

    # Create environment function for MAPPO that accepts seed parameter
    def env_func(seed=None, **kwargs):
        return create_env(multiagent=multiagent, gui=False, record_video=False, 
                         num_drones=num_drones if multiagent else 1, seed=seed)

    # Test the environment function
    print("Testing environment function...")
    test_env = env_func()
    print(f"✓ Environment created successfully")
    print(f"  Observation space: {test_env.observation_space}")
    print(f"  Action space: {test_env.action_space}")
    test_env.close()

    # Check GPU availability
    use_gpu_flag = use_gpu and torch.cuda.is_available()
    if use_gpu_flag:
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        if use_gpu:
            print("⚠ GPU requested but not available, using CPU")
        else:
            print("ℹ Using CPU (GPU disabled)")
    
    print(f"✓ Vectorized training configuration:")
    print(f"  Parallel environments: {rollout_batch_size}")
    print(f"  Number of workers: {num_workers}")
    print(f"  This will speed up training by ~{rollout_batch_size}x")
    
    # Custom MAPPO configuration
    mappo_config = MAPPO_CONFIG.copy()
    mappo_config.update({
        'output_dir': filename,
        'checkpoint_path': os.path.join(filename, 'model_latest.pt'),
        'max_env_steps': int(4e6) if local else int(1e4),  # Reduced for testing
        'eval_interval': 5000,
        'eval_batch_size': 3,
        'log_interval': 1,
        'save_interval': 20000,
        'early_save_interval': 5000,  # Save more frequently for safety
        'tensorboard': use_wandb,
        # MAPPO-specific parameters
        'centralized_critic': True,  # Enable centralized critic for MAPPO
        'share_actor_weights': True,  # Share actor parameters across agents
        'include_actions_in_critic': False,  # Don't include actions in critic input
        'global_state_dim': None,  # Auto-detect from environment
        # Training parameters
        'hidden_dim': 256,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'rollout_steps': 256,  # Reduced for testing
        'rollout_batch_size': rollout_batch_size,  # Number of parallel environments
        'num_workers': num_workers,  # Number of parallel processes for vectorized envs
        'opt_epochs': 10,  # Reduced for testing
        'mini_batch_size': 32,
        'use_gae': True,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        'clip_param': 0.1,
        'target_kl': 0.01,
        'entropy_coef': 0.005,
        'use_clipped_value': False,
        # Normalization
        'norm_obs': True,
        'norm_reward': False,
        'clip_obs': 10,
        'clip_reward': 10,
        'action_scale': 0.25,
    })

    # Log configuration to WandB
    if use_wandb:
        wandb_run.config.update(mappo_config)

    print("\n" + "="*60)
    print("MAPPO CONFIGURATION:")
    print("="*60)
    for key, value in {k: v for k, v in mappo_config.items() if not k.startswith('_')}.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            print(f"  {key}: {value}")
    print("="*60 + "\n")

    # Initialize MAPPO controller with GPU support
    print("Initializing MAPPO controller...")
    try:
        mappo_controller = MAPPO(
            env_func=env_func,
            training=True,
            use_gpu=use_gpu_flag,  # Enable GPU if available
            **mappo_config
        )
        print("✓ MAPPO controller initialized successfully")
        print(f"  Device: {mappo_controller.device}")
        print(f"  Vectorized environments: {mappo_controller.is_vectorized if hasattr(mappo_controller, 'is_vectorized') else 'N/A'}")
        if hasattr(mappo_controller, 'num_envs'):
            print(f"  Parallel environments: {mappo_controller.num_envs}")
    except Exception as e:
        print(f"✗ Error initializing MAPPO controller: {e}")
        import traceback
        traceback.print_exc()
        return

    # Reset the environment
    print("Resetting environment...")
    mappo_controller.reset()
    print("✓ Environment reset")

    # Training
    print("\n" + "="*60)
    print("STARTING MAPPO TRAINING")
    print("="*60)
    
    # Initialize model paths before try block to avoid UnboundLocalError
    final_model_path = os.path.join(filename, 'model_final.pt')
    
    try:
        mappo_controller.learn()
        print("✓ Training completed successfully!")
        
        # Save final model on successful completion
        try:
            mappo_controller.save(final_model_path)
            print(f"✓ Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"✗ Error saving final model: {e}")
            
    except KeyboardInterrupt:
        # Interruption is handled inside learn(), but ensure we close properly
        print("⚠ Training interrupted by user!")
        # Model should already be saved by the signal handler in learn()
        # Try to save final model if it wasn't saved yet
        try:
            if not os.path.exists(final_model_path):
                mappo_controller.save(final_model_path)
                print(f"✓ Final model saved on interruption to: {final_model_path}")
        except Exception as e:
            print(f"✗ Error saving on interruption: {e}")
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Try to save model even if training failed
        try:
            if hasattr(mappo_controller, 'total_steps') and mappo_controller.total_steps > 0:
                error_model_path = os.path.join(filename, f'model_error_step_{mappo_controller.total_steps}.pt')
                mappo_controller.save(error_model_path)
                print(f"✓ Model saved after error to: {error_model_path}")
                # Also try to save as final model
                if not os.path.exists(final_model_path):
                    mappo_controller.save(final_model_path)
                    print(f"✓ Final model also saved to: {final_model_path}")
        except Exception as save_error:
            print(f"✗ Error saving model after failure: {save_error}")
    
    finally:
        # Ensure environments are closed
        try:
            mappo_controller.close()
        except:
            pass

    ############################################################
    # Evaluation and Demonstration
    ############################################################

    # Load the best model for evaluation
    best_model_path = os.path.join(filename, 'model_best.pt')
    model_to_load = None
    
    if os.path.exists(best_model_path):
        print("\nLoading best model for evaluation...")
        model_to_load = best_model_path
    elif os.path.exists(final_model_path):
        print("\nBest model not found, using final model...")
        model_to_load = final_model_path
    else:
        print("\nNo saved models found. Skipping evaluation.")
        if use_wandb:
            wandb_run.finish()
        return

    try:
        mappo_controller.load(model_to_load)
        print(f"✓ Model loaded from: {model_to_load}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Create test environment
    print("\nCreating test environment...")
    test_env = create_env(
        multiagent=multiagent, 
        gui=gui, 
        record_video=record_video,
        num_drones=num_drones if multiagent else 1
    )

    # Setup logger
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=num_drones if multiagent else 1,
        output_folder=output_folder,
        colab=colab,
        use_wandb=use_wandb,
        wandb_run=wandb_run
    )

    # Run evaluation
    print("\nRunning evaluation...")
    try:
        eval_results = mappo_controller.run(
            env=test_env, 
            n_episodes=3,  # Reduced for faster testing
            render=False  # Don't render during batch evaluation
        )
        
        if eval_results and 'ep_returns' in eval_results:
            mean_reward = eval_results['ep_returns'].mean()
            std_reward = eval_results['ep_returns'].std()
            mean_length = eval_results['ep_lengths'].mean()
            std_length = eval_results['ep_lengths'].std()
            
            print(f"\n" + "="*60)
            print("EVALUATION RESULTS:")
            print("="*60)
            print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode lengths: {mean_length:.2f} +/- {std_length:.2f}")
            print("="*60)

            # Log evaluation results to WandB
            if use_wandb:
                wandb_run.log({
                    "final_evaluation/mean_reward": mean_reward,
                    "final_evaluation/std_reward": std_reward,
                    "final_evaluation/mean_episode_length": mean_length,
                    "final_evaluation/std_episode_length": std_length,
                })
        else:
            print("⚠ Evaluation returned no results")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")

    # Demonstration run with rendering
    print("\n" + "="*60)
    print("RUNNING DEMONSTRATION")
    print("="*60)
    
    try:
        obs, info = test_env.reset(seed=42)
        start = time.time()
        
        # Run for a shorter period for demonstration
        max_steps = min((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ, 500)
        
        total_reward = 0
        episode_count = 0
        
        for i in range(max_steps):
            try:
                action = mappo_controller.select_action(obs, info)
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                total_reward += reward if isinstance(reward, (int, float)) else np.mean(reward)
                
                # Log data for visualization
                obs_flat = obs.squeeze()
                action_flat = action.squeeze()
                
                if DEFAULT_OBS == ObservationType.KIN:
                    if not multiagent:
                        logger.log(
                            drone=0,
                            timestamp=i / test_env.CTRL_FREQ,
                            state=np.hstack([
                                obs_flat[0:3] if obs_flat.shape[0] >= 3 else np.zeros(3),        # position
                                np.zeros(4),          # orientation (quat) - not used
                                obs_flat[3:15] if obs_flat.shape[0] >= 15 else np.zeros(12),     # velocity, angular velocity, etc.
                                action_flat           # action
                            ]),
                            control=np.zeros(12)      # control signals
                        )
                    else:
                        for d in range(min(num_drones, obs_flat.shape[0] if isinstance(obs_flat, np.ndarray) else 1)):
                            drone_obs = obs_flat[d] if isinstance(obs_flat, np.ndarray) and len(obs_flat.shape) > 1 else obs_flat
                            logger.log(
                                drone=d,
                                timestamp=i / test_env.CTRL_FREQ,
                                state=np.hstack([
                                    drone_obs[0:3] if len(drone_obs) >= 3 else np.zeros(3),    # position
                                    np.zeros(4),         # orientation
                                    drone_obs[3:15] if len(drone_obs) >= 15 else np.zeros(12),   # velocity, angular velocity
                                    action_flat[d] if isinstance(action_flat, np.ndarray) and len(action_flat.shape) > 1 else action_flat  # action
                                ]),
                                control=np.zeros(12)     # control signals
                            )
                
                # Render and sync
                if gui:
                    test_env.render()
                
                if i % 50 == 0:
                    print(f"Step: {i}, Reward: {reward:.3f}")
                
                # Log to WandB periodically
                if use_wandb and i % 10 == 0:
                    wandb_run.log({
                        "demonstration/reward": reward if isinstance(reward, (int, float)) else np.mean(reward),
                        "demonstration/step": i
                    })
                
                sync(i, start, test_env.CTRL_TIMESTEP)
                
                if terminated or truncated:
                    print(f"Episode {episode_count + 1} finished! Total reward: {total_reward:.2f}")
                    episode_count += 1
                    total_reward = 0
                    obs, info = test_env.reset(seed=42 + episode_count)
                    
                    if episode_count >= 2:  # Run 2 episodes for demonstration
                        break
            
            except Exception as step_error:
                print(f"✗ Error in step {i}: {step_error}")
                break
        
        print(f"\nDemonstration completed. Ran {episode_count} episode(s).")
        
    except Exception as e:
        print(f"✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        test_env.close()
        print("✓ Test environment closed")

    # Generate plots
    if plot and DEFAULT_OBS == ObservationType.KIN:
        print("\nGenerating plots...")
        try:
            logger.plot()
            print("✓ Plots generated")
        except Exception as e:
            print(f"✗ Error generating plots: {e}")

    # Finish WandB run
    if use_wandb:
        try:
            wandb_run.finish()
            print("✓ WandB run finished")
        except Exception as e:
            print(f"✗ Error finishing WandB run: {e}")

    print("\n" + "="*60)
    print("MAPPO TRAINING AND EVALUATION COMPLETED!")
    print(f"Results saved to: {filename}")
    print("="*60)

def test_mappo_components():
    """Test function to verify MAPPO components are working correctly."""
    print("\n" + "="*60)
    print("TESTING MAPPO COMPONENTS")
    print("="*60)
    
    try:
        # Create a simple test environment
        import gymnasium as gym
        test_env = gym.make('Pendulum-v1')
        
        # Test buffer
        print("Testing MAPPOBuffer...")
        buffer = MAPPOBuffer(
            test_env.observation_space,
            test_env.action_space,
            max_length=10,
            batch_size=2
        )
        print("✓ MAPPOBuffer initialized successfully")
        
        # Test with multi-agent configuration
        print("\nTesting multi-agent configuration...")
        # Create a mock multi-agent observation space
        from gymnasium.spaces import Box
        multi_obs_space = Box(low=-np.inf, high=np.inf, shape=(2, 3))  # 2 agents, 3-dim obs each
        multi_act_space = Box(low=-1, high=1, shape=(2, 1))  # 2 agents, 1-dim action each
        
        multi_buffer = MAPPOBuffer(
            multi_obs_space,
            multi_act_space,
            max_length=10,
            batch_size=1,
            include_global_state=True
        )
        print("✓ Multi-agent buffer initialized successfully")
        
        # Test agent
        print("\nTesting MAPPOAgent...")
        agent = MAPPOAgent(
            test_env.observation_space,
            test_env.action_space,
            hidden_dim=256,
            centralized_critic=True
        )
        print("✓ MAPPOAgent initialized successfully")
        
        # Test with multi-agent
        multi_agent = MAPPOAgent(
            multi_obs_space,
            multi_act_space,
            hidden_dim=256,
            centralized_critic=True,
            share_actor_weights=True
        )
        print("✓ Multi-agent MAPPOAgent initialized successfully")
        
        # Test MAPPO controller
        print("\nTesting MAPPO controller...")
        def dummy_env_func():
            return test_env
        
        # Create minimal config
        config = {
            'max_env_steps': 100,
            'rollout_steps': 10,
            'rollout_batch_size': 1,
            'output_dir': './test_output',
            'checkpoint_path': './test_output/model.pt'
        }
        
        controller = MAPPO(
            env_func=dummy_env_func,
            training=True,
            **config
        )
        print("✓ MAPPO controller initialized successfully")
        
        test_env.close()
        
        print("\n" + "="*60)
        print("✓ ALL MAPPO COMPONENTS WORKING CORRECTLY!")
        print("="*60)
        
        # Clean up
        import shutil
        if os.path.exists('./test_output'):
            shutil.rmtree('./test_output')
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing MAPPO components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_compatibility():
    """Test if the environment is compatible with MAPPO."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT COMPATIBILITY")
    print("="*60)
    
    try:
        # Test single agent environment
        print("\nTesting single agent environment...")
        env = HoverAviary(gui=False, obs=DEFAULT_OBS, act=DEFAULT_ACT)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation shape: {env.observation_space.shape}")
        print(f"  Action shape: {env.action_space.shape}")
        env.close()
        print("✓ Single agent environment compatible")
        
        # Test multi-agent environment
        print("\nTesting multi-agent environment...")
        env = MultiHoverAviary(gui=False, num_drones=2, obs=DEFAULT_OBS, act=DEFAULT_ACT)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation shape: {env.observation_space.shape}")
        print(f"  Action shape: {env.action_space.shape}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"  Reset observation shape: {obs.shape}")
        action = np.zeros(env.action_space.shape)
        obs, reward, done, info = env.step(action)
        print(f"  Step observation shape: {obs.shape}")
        print(f"  Reward type: {type(reward)}, shape: {reward.shape if hasattr(reward, 'shape') else 'scalar'}")
        env.close()
        print("✓ Multi-agent environment compatible")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Environment compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='MAPPO training script for drone control')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use multi-agent environment (default: True)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--use_wandb',          default=DEFAULT_USE_WANDB,     type=str2bool,      help='Whether to use Weights & Biases for logging (default: True)', metavar='')
    parser.add_argument('--wandb_project',      default=DEFAULT_WANDB_PROJECT, type=str,           help='WandB project name (default: "gym-pybullet-drones-mappo")', metavar='')
    parser.add_argument('--wandb_entity',       default=DEFAULT_WANDB_ENTITY,  type=str,           help='WandB entity (team name) (default: None)', metavar='')
    parser.add_argument('--test_components',    default=False,                 type=str2bool,      help='Test MAPPO components before training (default: False)', metavar='')
    parser.add_argument('--test_env',           default=False,                 type=str2bool,      help='Test environment compatibility before training (default: False)', metavar='')
    parser.add_argument('--num_drones',         default=DEFAULT_AGENTS,        type=int,           help='Number of drones for multi-agent (default: 2)', metavar='')
    parser.add_argument('--rollout_batch_size', default=8,                     type=int,           help='Number of parallel environments for vectorized training (default: 8)', metavar='')
    parser.add_argument('--num_workers',        default=4,                     type=int,           help='Number of parallel processes for vectorized environments (default: 4)', metavar='')
    parser.add_argument('--use_gpu',            default=True,                  type=str2bool,      help='Use GPU if available (default: True)', metavar='')
    
    ARGS = parser.parse_args()
    
    # Update DEFAULT_AGENTS if specified
    if ARGS.num_drones != DEFAULT_AGENTS:
        DEFAULT_AGENTS = ARGS.num_drones
    
    # Test environment if requested
    if ARGS.test_env:
        if test_environment_compatibility():
            print("\nEnvironment compatibility test passed!")
        else:
            print("\nEnvironment compatibility test failed!")
            sys.exit(1)
    
    # Test components if requested
    if ARGS.test_components:
        if test_mappo_components():
            print("\nComponent testing completed successfully!")
        else:
            print("\nComponent testing failed!")
            sys.exit(1)
    
    # Remove test arguments before passing to run()
    args_dict = vars(ARGS)
    args_dict.pop('test_components', None)
    args_dict.pop('test_env', None)
    # Keep num_drones, rollout_batch_size, num_workers, use_gpu for run()
    
    # Run main training
    try:
        run(**args_dict)
    except KeyboardInterrupt:
        print("\n\n⚠ Script interrupted by user!")
    except Exception as e:
        print(f"\n\n✗ Script failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
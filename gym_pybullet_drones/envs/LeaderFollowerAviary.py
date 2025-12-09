import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class LeaderFollowerAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        The reward is the sum of:
        1. Negative squared distance for the leader (drone 0) to hover at (0, 0, 0.5).
        2. Negative squared distance for followers (drones 1 to N-1) to match the leader's Z height while maintaining their current X/Y position.

        Returns
        -------
        float
            The total scalar reward (sum of all individual rewards).

        """
        total_reward = 0
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Leader reward (Drone 0): Hover at (0, 0, 0.5)
        total_reward += -1 * np.linalg.norm(np.array([0, 0, 0.5]) - states[0, 0:3])**2
        
        # Follower rewards (Drone 1 to N-1)
        for i in range(1, self.NUM_DRONES):
            # Target for follower i: (x_i, y_i, z_leader)
            target_pos = np.array([states[i, 0], states[i, 1], states[0, 2]])
            # Follower reward: -(1/N) * L2_norm(target_pos - pos_i)^2
            follower_reward = -(1/self.NUM_DRONES) * np.linalg.norm(target_pos - states[i, 0:3])**2
            total_reward += follower_reward

        return total_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value.

        Returns
        -------
        bool
            Whether the current episode is done due to failure/success.
        """
        # In this task, there is no explicit success criteria leading to termination
        # other than potentially reaching a very small error, which is usually handled
        # by the time limit.
        return False

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out or went out of bounds.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Check for out of bounds or excessive tilt (using MultiHoverAviary bounds for consistency)
        MAX_DISTANCE = 2.0
        MAX_TILT = 0.4
        
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > MAX_DISTANCE or abs(states[i][1]) > MAX_DISTANCE or states[i][2] > MAX_DISTANCE # Truncate when a drone is too far away
             or abs(states[i][7]) > MAX_TILT or abs(states[i][8]) > MAX_TILT # Truncate when a drone is too tilted
            ):
                return True
                
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}
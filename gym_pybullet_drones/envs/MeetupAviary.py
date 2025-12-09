import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MeetupAviary(BaseRLAviary):
    """Multi-agent RL problem: meet mid-flight."""

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

        The reward is based on the negative squared distance between paired drones.

        Returns
        -------
        float
            The total scalar reward (sum of all pairing rewards).

        """
        total_reward = 0
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Original logic pairs drone 0 with N-1, 1 with N-2, etc.
        for i in range(int(self.NUM_DRONES/2)):
            # Calculate negative squared distance between drone i and its partner
            pair_reward = -1 * np.linalg.norm(states[i, 0:3] - states[self.NUM_DRONES-1-i, 0:3])**2
            total_reward += pair_reward * 2 # Multiply by 2 because the original code assigned the reward to both drones in the pair

        return total_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value.

        The episode terminates if all pairs meet (distance below a threshold).

        Returns
        -------
        bool
            Whether the current episode is done due to success.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Check if all pairs are within a small meeting distance (e.g., 0.1 meters)
        MEET_DISTANCE_THRESHOLD = 0.1
        
        for i in range(int(self.NUM_DRONES/2)):
            distance = np.linalg.norm(states[i, 0:3] - states[self.NUM_DRONES-1-i, 0:3])
            if distance > MEET_DISTANCE_THRESHOLD:
                return False # If any pair is too far, don't terminate
        
        # If the loop finishes, all pairs have met
        return True

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        The episode is truncated if the time limit is reached or a drone goes out of bounds.

        Returns
        -------
        bool
            Whether the current episode timed out or went out of bounds.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Bounding Box (e.g., 5 meters in XY, 3 meters in Z)
        MAX_X_Y = 5.0
        MAX_Z = 3.0
        
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > MAX_X_Y or abs(states[i][1]) > MAX_X_Y or states[i][2] > MAX_Z # Truncate when a drone is too far away
             or states[i][2] < 0.1 # Truncate if drone hits the ground
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted (roll/pitch > 0.4 rad)
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
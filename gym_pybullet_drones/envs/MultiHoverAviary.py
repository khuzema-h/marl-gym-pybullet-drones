import numpy as np
import torch

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiHoverAviary(BaseRLAviary):
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
                 act: ActionType=ActionType.RPM
                 ):
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
            Whether to save a video of the simulation.
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
        self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(num_drones)])
        # + np.array([[0,0,1/(i+1)] for i in range(num_drones)])

    def reset(self, seed : int = None, options : dict = None):
        #  Add a small random noise to the initial positions and call the parent reset method
        # First check if the original initial position has been saved
        if not hasattr(self, 'ORIGINAL_INIT_XYZS'):
            self.ORIGINAL_INIT_XYZS = self.INIT_XYZS.copy()

        # Create a new array of initial positions with random noise
        # Add +/- 0.25 to the original x and y positions, and +/- 0.1 to the z position 
        new_init_xyzs = self.ORIGINAL_INIT_XYZS.copy() + np.random.uniform(-0.25, 0.25, (self.NUM_DRONES, 3))
        new_init_xyzs[:, 2] = np.clip(new_init_xyzs[:, 2], 0.1, 1.0)

        # Make sure the drones are not too close to each other (collision) AND not too low
        # We loop until both conditions are satisfied
        valid_config = False
        while not valid_config:
            # Check collisions
            dists = np.linalg.norm(new_init_xyzs[:, np.newaxis, :] - new_init_xyzs[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf) # Ignore self-collision
            collision = np.any(dists < 0.5)
            # Check too low
            too_low = np.any(new_init_xyzs[:, 2] < 0.1)
            
            if not collision and not too_low:
                valid_config = True
            else:
                # Regenerate if invalid
                new_init_xyzs = self.ORIGINAL_INIT_XYZS.copy() + np.random.uniform(-0.25, 0.25, (self.NUM_DRONES, 3))
                new_init_xyzs[:, 2] = np.clip(new_init_xyzs[:, 2], 0.1, 1.0)

        # Update the initial positions and target positions 
        self.INIT_XYZS = new_init_xyzs
        self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(self.NUM_DRONES)])

        # Initialize termination reason
        self.termination_reasons = []
        return super().reset(seed=seed, options=options)

    ################################################################################
    
    # def _computeReward(self):
    #     """Computes the current reward value.

    #     Returns
    #     -------
    #     float
    #         The reward.

    #     """
    #     states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     ret = 0
    #     for i in range(self.NUM_DRONES):
    #         ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])**4)
    #     return ret
    def _computeReward(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        reward = 0.0

        for i in range(self.NUM_DRONES):
            pos = states[i, 0:3]
            vel = states[i, 10:13]
            target = self.TARGET_POS[i]

            # ------------------------
            # Errors
            # ------------------------
            err_xy = np.linalg.norm(pos[0:2] - target[0:2])
            err_z  = pos[2] - target[2]
            vel_z  = vel[2]

            # ------------------------
            # Position shaping
            # ------------------------
            # r_xy = np.exp(-2.0 * err_xy)
            r_xy = 1.0 / (1 + err_xy)
            r_z  = np.exp(-7.5 * abs(err_z))   # Stronger ascent gradient

            # ------------------------
            # Velocity damping (ONLY near target)
            # ------------------------
            if abs(err_z) < 0.2:
                r_vel = -1.5 * vel_z**2
            else:
                r_vel = 0.0

            # ------------------------
            # Soft overshoot penalty (asymmetric)
            # ------------------------
            height_penalty = 0.0
            if err_z > 0.0:
                height_penalty = -1.0 * err_z**2

            # ------------------------
            # Hover bonus (tight attractor)
            # ------------------------
            hover_bonus = 0.0
            if err_xy < 0.03 and abs(err_z) < 0.03 and abs(vel_z) < 0.03:
                hover_bonus = 0.5

            reward += (
                r_xy +
                r_z +
                r_vel +
                # height_penalty +
                hover_bonus
            )

        # ------------------------
        # MAPPO-correct normalization
        # ------------------------
        reward /= self.NUM_DRONES

        return float(reward)








    ################################################################################
    
    # def _computeTerminated(self):
    #     """Computes the current done value.

    #     Returns
    #     -------
    #     bool
    #         Whether the current episode is done.

    #     """
    #     states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     dist = 0
    #     for i in range(self.NUM_DRONES):
    #         dist += np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
    #     if dist < .0001:
    #         return True
    #     else:
    #         return False
        # return terminated
    
    def _computeTerminated(self):
        terminated = False
        reasons = []

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            x, y, z = state[0], state[1], state[2]
            roll, pitch = state[7], state[8]

            # 💥 Crash into ground
            if z < 0.03:
                terminated = True
                reasons.append(f"Drone {i} crashed (z={z:.2f})")

            # 💥 Completely flipped (hard failure)
            if abs(roll) > 1.2 or abs(pitch) > 1.2:
                terminated = True
                reasons.append(f"Drone {i} flipped (roll={roll:.2f}, pitch={pitch:.2f})")

            # 💥 Escaped XY boundary (safety)
            if abs(x) > 3.0 or abs(y) > 3.0:
                terminated = True
                reasons.append(f"Drone {i} out of bounds (pos=[{x:.2f}, {y:.2f}, {z:.2f}])")

        self.termination_reasons = reasons
        return terminated




    ################################################################################
    
    # def _computeTruncated(self):
    #     """Computes the current truncated value.

    #     Returns
    #     -------
    #     bool
    #         Whether the current episode timed out.

    #     """
    #     states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     for i in range(self.NUM_DRONES):
    #         if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
    #          or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
    #         ):
    #             return True
    #     if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
    #         return True
    #     else:
    #         return False
    def _computeTruncated(self):
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC



    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42, "termination_reasons": self.termination_reasons} #### Calculated by the Deep Thought supercomputer in 7.5M years
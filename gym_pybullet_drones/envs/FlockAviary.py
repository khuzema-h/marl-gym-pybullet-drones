import math
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class FlockAviary(BaseRLAviary):
    """Multi-agent RL problem: flocking."""

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
        """Computes the current reward value(s).

        Returns
        -------
        float
            The reward value (sum of all drones' rewards).

        """
        # Parse state to get position and velocity
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # pos: (1, NUM_DRONES, 3), vel: (1, NUM_DRONES, 3) to match logic
        pos = np.zeros((1, self.NUM_DRONES, 3))
        vel = np.zeros((1, self.NUM_DRONES, 3))
        
        for i in range(self.NUM_DRONES):
            pos[0, i, :] = states[i][0:3]
            vel[0, i, :] = states[i][10:13]

        # --- Compute Metrics ---
        
        # 1. Velocity alignment
        ali = 0
        EPSILON = 1e-3
        linear_vel_norm = np.linalg.norm(vel, axis=2)
        for i in range(self.NUM_DRONES):
            for j in range(self.NUM_DRONES):
                if j != i:
                    d = np.einsum('ij,ij->i', vel[:, i, :], vel[:, j, :])
                    ali += (d / (linear_vel_norm[:, i] + EPSILON) / (linear_vel_norm[:, j] + EPSILON))
        
        if self.NUM_DRONES > 1:
            ali /= (self.NUM_DRONES * (self.NUM_DRONES - 1))
        else:
            ali = np.array([0.0])

        # 2. Flocking speed
        cof_v = np.mean(vel, axis=1)  # center of flock speed
        avg_flock_linear_speed = np.linalg.norm(cof_v, axis=-1)

        # 3. Spacing
        avg_flock_spac_rew = 0.0
        var_flock_spacing = np.array([0.0])
        
        if self.NUM_DRONES > 1:
            whole_flock_spacing = []
            for i in range(self.NUM_DRONES):
                flck_neighbor_pos = np.delete(pos, [i], 1)
                drone_neighbor_pos_diff = flck_neighbor_pos - np.reshape(pos[:, i, :], (pos[:, i, :].shape[0], 1, -1))
                drone_neighbor_dis = np.linalg.norm(drone_neighbor_pos_diff, axis=-1)
                drone_spacing = np.amin(drone_neighbor_dis, axis=-1)
                whole_flock_spacing.append(drone_spacing)
            
            whole_flock_spacing = np.stack(whole_flock_spacing, axis=-1)
            avg_flock_spacing = np.mean(whole_flock_spacing, axis=-1)
            var_flock_spacing = np.var(whole_flock_spacing, axis=-1)

            FLOCK_SPACING_MIN = 1.0
            FLOCK_SPACING_MAX = 3.0
            
            # Calculate spacing penalty
            if FLOCK_SPACING_MIN < avg_flock_spacing[0] < FLOCK_SPACING_MAX:
                avg_flock_spac_rew = 0.0
            else:
                avg_flock_spac_rew = min(math.fabs(avg_flock_spacing[0] - FLOCK_SPACING_MIN),
                                         math.fabs(avg_flock_spacing[0] - FLOCK_SPACING_MAX))

        # Combine components
        # We sum the alignment and speed, and subtract spacing penalty and variance
        total_reward = ali[0] + avg_flock_linear_speed[0] - avg_flock_spac_rew - var_flock_spacing[0]
        
        return total_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value.

        Returns
        -------
        bool
            Whether the current episode is done due to failure/success.

        """
        # In flocking, we usually don't have a "target" to hit, 
        # so we just return False unless you want to add collision checks.
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
        for i in range(self.NUM_DRONES):
            # Truncate if a drone is too far away (e.g. 10 meters) or tilted too much
            if (abs(states[i][0]) > 10.0 or abs(states[i][1]) > 10.0 or states[i][2] > 10.0 
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Tilt limit
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
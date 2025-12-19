import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class SpiralFormationAviary(BaseRLAviary):
    """
    Multi-agent spiral formation environment with **explicit spiral kinematics**.

    Key properties:
    - ActionType.VEL (vx, vy, vz, yaw_rate)
    - Analytic spiral target position AND velocity
    - Rewards dominated by velocity tracking + tangential motion
    - No hover–jitter local minimum
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 3,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 spiral_radius: float = 0.4,
                 spiral_period: float = 10.0,
                 height_rate: float = 0.05,
                 target_center: np.ndarray = np.array([0.0, 0.0, 0.0])
                 ):

        self.EPISODE_LEN_SEC = 12

        self.R = spiral_radius
        self.PERIOD = spiral_period
        self.OMEGA = 2 * np.pi / self.PERIOD
        self.VZ = height_rate
        self.CENTER = target_center

        if initial_xyzs is None:
            initial_xyzs = np.array([
                [self.R * np.cos(2 * np.pi * i / num_drones),
                 self.R * np.sin(2 * np.pi * i / num_drones),
                 0.3]
                for i in range(num_drones)
            ])

        if initial_rpys is None:
            initial_rpys = np.zeros((num_drones, 3))

        self._obs_type = obs
        self._num_drones = num_drones

        super().__init__(
            drone_model=drone_model,
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

        self.observation_space = self._observationSpace()

        print(f"[SpiralFormationAviary] Initialized | obs shape = {self.observation_space.shape}")

    # ------------------------------------------------------------------

    def _spiral_reference(self, drone_id: int):
        """Return target position and velocity for drone_id."""
        t = self.step_counter / self.PYB_FREQ
        phase = self.OMEGA * t + 2 * np.pi * drone_id / self.NUM_DRONES

        # Position
        x = self.CENTER[0] + self.R * np.cos(phase)
        y = self.CENTER[1] + self.R * np.sin(phase)
        z = 0.3 + self.VZ * t
        pos_ref = np.array([x, y, z])

        # Velocity (analytic derivative)
        vx = -self.R * self.OMEGA * np.sin(phase)
        vy =  self.R * self.OMEGA * np.cos(phase)
        vz = self.VZ
        vel_ref = np.array([vx, vy, vz])

        return pos_ref, vel_ref, phase

    # ------------------------------------------------------------------

    def _observationSpace(self):
        if self._obs_type == ObservationType.KIN:
            base_low = np.full((self.NUM_DRONES, 108), -np.inf, dtype=np.float32)
            base_high = np.full((self.NUM_DRONES, 108), np.inf, dtype=np.float32)

            # Extra: rel_pos(3), rel_vel(3), sinφ, cosφ, vel_ref(3)
            ext_low = np.full((self.NUM_DRONES, 11), -np.inf, dtype=np.float32)
            ext_high = np.full((self.NUM_DRONES, 11), np.inf, dtype=np.float32)

            low = np.concatenate([base_low, ext_low], axis=1)
            high = np.concatenate([base_high, ext_high], axis=1)
            return spaces.Box(low=low, high=high, dtype=np.float32)

        return super()._observationSpace()

    # ------------------------------------------------------------------

    def _computeObs(self):
        obs = super()._computeObs()

        if self.OBS_TYPE != ObservationType.KIN:
            return obs

        augmented = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            vel = state[3:6]

            pos_ref, vel_ref, phase = self._spiral_reference(i)

            rel_pos = pos_ref - pos
            rel_vel = vel_ref - vel

            extra = np.concatenate([
                rel_pos,
                rel_vel,
                np.array([np.sin(phase), np.cos(phase)]),
                vel_ref
            ])

            augmented.append(np.concatenate([obs[i], extra]))

        return np.array(augmented, dtype=np.float32)

    # ------------------------------------------------------------------

    def _computeReward(self):
        reward = 0.0

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            vel = state[3:6]

            pos_ref, vel_ref, _ = self._spiral_reference(i)

            # Position tracking
            r_pos = np.exp(-4.0 * np.linalg.norm(pos - pos_ref) ** 2)

            # Velocity tracking (dominant)
            r_vel = np.exp(-2.0 * np.linalg.norm(vel - vel_ref) ** 2)

            # Tangential velocity encouragement
            r_xy = pos[0:2] - self.CENTER[0:2]
            if np.linalg.norm(r_xy) > 1e-3:
                radial = r_xy / np.linalg.norm(r_xy)
                tangent = np.array([-radial[1], radial[0]])
                v_xy = vel[0:2]
                if np.linalg.norm(v_xy) > 1e-3:
                    r_tan = max(0.0, np.dot(v_xy / np.linalg.norm(v_xy), tangent))
                else:
                    r_tan = 0.0
            else:
                r_tan = 0.0

            reward += 1.0 * r_pos + 2.0 * r_vel + 1.0 * r_tan

        return reward / self.NUM_DRONES

    # ------------------------------------------------------------------

    def _computeTerminated(self):
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            z = state[2]
            if z < 0.05 or z > 3.0:
                return True
        return False

    # ------------------------------------------------------------------

    def _computeTruncated(self):
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    # ------------------------------------------------------------------

    def _computeInfo(self):
        return {
            "time": self.step_counter / self.PYB_FREQ,
            "omega": self.OMEGA,
            "radius": self.R
        }

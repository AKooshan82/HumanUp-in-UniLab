"""G1 getting-up environment: recover from a fallen (supine) pose to standing.

Ported from HumanUP's Stage-I ``g1waist`` task (Isaac Gym / legged_gym,
https://github.com/RunpeiDong/HumanUP) onto UniLab's MuJoCo backend. The
fallen-pose keyframe and reward set are a pragmatic port, not a faithful
reproduction: HumanUP's per-step assistive-force/domain-rand machinery and
its head-height reward (no head body exists on this repo's G1 asset) are
dropped, and reward weights are an untuned starting point.

Adapted from HumanUP's ``envs/g1waist/g1waist_up.py`` and
``envs/g1waistroll/g1waistroll_up.py`` (He, Dong, Chen, Gupta, RSS 2025;
Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv


@dataclass
class G1GetupRewardConfig:
    scales: dict[str, float]
    base_height_target: float = 0.754


@registry.envcfg("G1Getup")
@dataclass
class G1GetupCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_getup.xml")
        )
    )
    max_episode_seconds: float | None = 10.0
    # HumanUP found getting-up specifically needs ~1kHz physics (vs the usual
    # 150Hz used for G1 locomotion tasks in this repo) to transfer; ctrl_dt
    # unchanged (still 50Hz actions -> 20 substeps instead of 3).
    sim_dt: float = 0.001
    ctrl_dt: float = 0.02
    reward_config: G1GetupRewardConfig | None = None
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    # Which scene keyframe to reset every episode to. "supine" (face up) is
    # HumanUP's g1waist task; G1RollOverCfg below overrides this to "prone"
    # (face down) to reproduce HumanUP's companion g1waistroll task, reusing
    # the same env/reward code with a different fixed starting pose.
    fallen_keyframe: str = "supine"


@registry.envcfg("G1RollOver")
@dataclass
class G1RollOverCfg(G1GetupCfg):
    fallen_keyframe: str = "prone"


class G1GetupDomainRandomizationProvider(LocomotionDRProvider):
    """Always resets to the fixed supine keyframe, matching HumanUP's g1waist.

    HumanUP applies no orientation/joint-angle randomization to its fallen
    pose (see docs/porting notes) -- this mirrors that rather than reusing
    the yaw-randomized reset every other locomotion task in this repo uses.
    """

    def __init__(self, *, supine_qpos: np.ndarray):
        self._supine_qpos = supine_qpos

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(self._supine_qpos, (num_reset, 1))
        qvel = np.zeros((num_reset, env._init_qvel.shape[0]), dtype=get_global_dtype())
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3])
        info_updates: dict[str, Any] = {
            "commands": np.zeros((num_reset, 3), dtype=get_global_dtype()),
            "current_actions": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "last_actions": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)  # type: ignore[no-any-return]


class G1GetupEnv(G1BaseEnv):
    _cfg: G1GetupCfg
    _reward_cfg: Any

    def __init__(self, cfg: G1GetupCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()

        fallen_qpos = np.asarray(
            backend.get_keyframe_qpos(cfg.fallen_keyframe), dtype=get_global_dtype()
        )
        dr_provider = G1GetupDomainRandomizationProvider(supine_qpos=fallen_qpos)
        self._init_domain_randomization(dr_provider)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(29) + dof_vel(29) + action(29) = 93
        return {"obs": 93, "critic": 96}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "getup_height": self._reward_getup_height,
            "orientation": rewards.orientation,
            "upright": rewards.upright,
            "pose": rewards.similar_to_default,
            "action_rate": rewards.action_rate,
        }

    def _reward_getup_height(self, ctx: RewardContext) -> np.ndarray:
        """exp(clip(base_z, 0, target)) - 1 -- ported from HumanUP's base_height_exp.

        Unlike this repo's generic ``rewards.base_height`` (a squared penalty
        around a fixed target, meant for a robot that's already walking), this
        only ever rewards rising and saturates at the target height, which is
        what you want when the episode starts on the floor.
        """
        target = self._reward_cfg.base_height_target
        height = np.clip(ctx.base_height, 0.0, target)
        return np.asarray(np.exp(height) - 1.0, dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # No early termination on orientation/height, matching HumanUP: the
        # robot starts on the ground by design, so "fallen" isn't a failure
        # state here. Episodes end only via the framework's own
        # max_episode_seconds timeout (NpEnv._compute_truncated).
        terminated = np.zeros((self._num_envs,), dtype=np.bool_)

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel: np.ndarray, gyro: np.ndarray, gravity: np.ndarray,
        dof_pos: np.ndarray, dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        last_actions = info.get("current_actions", np.zeros_like(diff))

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)

        actor = np.concatenate(
            [noisy_gyro, -noisy_gravity, noisy_diff, noisy_dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, linvel],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": actor, "critic": critic}

    def _build_reward_context(
        self, info: dict, linvel: np.ndarray, gyro: np.ndarray, gravity: np.ndarray,
        dof_pos: np.ndarray, dof_vel: np.ndarray,
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
        )

    def _compute_reward(
        self, info: dict, linvel: np.ndarray, gyro: np.ndarray, gravity: np.ndarray,
        dof_pos: np.ndarray, dof_vel: np.ndarray,
    ) -> np.ndarray:
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )


registry.register_env("G1Getup", G1GetupEnv, sim_backend="mujoco")
registry.register_env("G1RollOver", G1GetupEnv, sim_backend="mujoco")

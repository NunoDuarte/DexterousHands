from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import os
import random
import torch
import torch.nn.parallel
from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task_qb import BaseTaskQb
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil


class QbHandPCD(BaseTaskQb):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.robot_position_noise = self.cfg["env"]["robotPositionNoise"]
        self.robot_rotation_noise = self.cfg["env"]["robotRotationNoise"]
        self.robot_dof_noise = self.cfg["env"]["robotDofNoise"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # Arm controller type
        self.arm_control_type = self.cfg["env"]["armControlType"]
        assert self.arm_control_type in {"osc", "pos", "vel"},\
            "Invalid control type specified. Must be one of: {osc, pos, vel}"

        # Hand controller type
        self.hand_control_type = self.cfg["env"]["handControlType"]
        assert self.hand_control_type in {"binary", "effort"},\
            "Invalid control type specified. Must be one of: {binary, effort}"

        # dimensions
        # obs include: object_pose (7) + eef_pose (7) + relative_object_eef_pos (3)
        self.full_state = 23
        self.pointCloudDownsampleNum = 768      # Point Cloud compression for observation space
        self.num_obs = self.full_state + self.pointCloudDownsampleNum * 3
        self.cfg["env"]["numObservations"] = self.num_obs

        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        # if arm_osc control: delta eef (6)
        # if arm_pos control: arm joint angles (7)
        # if hand_binary control: bool gripper (1)
        # if hand_pos control: finger joint angles (17)
        num_actions = 0
        if self.arm_control_type == "osc":
            num_actions += 6
        elif self.arm_control_type == "pos":
            num_actions += 7
        elif self.arm_control_type == "vel":
            num_actions += 7

        if self.hand_control_type == "binary" or "effort": num_actions += 1

        self.cfg["env"]["numActions"] = num_actions

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._vel_control = None         # Torque actions
        self._effort_control = None  # Torque actions
        self._global_indices = None  # Unique indices corresponding to all envs in flattened array
        self.up_axis = "z"

        super().__init__(cfg=self.cfg)

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # add a sphere for placing zone
        self.axes_geom = gymutil.AxesGeometry(0.1)
        sphere_pose = gymapi.Transform()
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.05, 15, 15, sphere_pose, color=(1, 0, 0))

        # Kinova + Seed defaults
        # robot_default_dof_pos = [0, 0.6, 0, 1, 0, -0.4, -1.5] + [0.0] * 33  # home position
        # robot_default_dof_pos = [0, 0, 0, 2.43, 0, -0.85, -1.5] + [0.0] * 33     # retracted home position
        robot_default_dof_pos = [0.19, 0.42, -1.39, 1.04, -1.00, 0.55, 0.00] + [0.0] * 33 # our default position
        # robot_default_dof_pos[36] = 1.57  # thumb opened (probably not a good idea)
        self.robot_default_dof_pos = to_torch(robot_default_dof_pos, device=self.device)

        # OSC Gains
        self.kp = to_torch([200.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([20.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
            self.arm_control_type == "osc" else self._robot_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        # set velocity control limits
        max_vel = 0.6727 * 2
        self.arm_velocity_limits_urdf = to_torch([max_vel, max_vel, max_vel, max_vel, max_vel, max_vel, max_vel],
                                                 device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../IsaacGymEnvs/assets")
        robot_asset_file = "urdf/qb_hand/urdf/arm_qbhand_iit.urdf"

        object_asset_file = self.asset_files_dict[self.object_type]

        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.u_hand_min = []
        self.u_hand_max = []
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self._robot_effort_limits = []

        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            self._robot_effort_limits.append(robot_dof_props['effort'][i])

        # arm properties
        robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["armature"][7:].fill(0.001)
        robot_dof_props["stiffness"][7:].fill(250.0)
        robot_dof_props["damping"][7:].fill(10.0)
        robot_dof_props["effort"][7:].fill(60.0)

        print("self.num_robot_bodies: ", self.num_robot_bodies)
        print("self.num_robot_shapes: ", self.num_robot_shapes)
        print("self.num_robot_dofs: ", self.num_robot_dofs)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_lower_limits[:7] = -3.14
        self.robot_dof_upper_limits[:7] = 3.14

        if self.hand_control_type == 'binary':
            self.u_hand_min = to_torch([0.0], device=self.device)
            self.u_hand_max = to_torch([0.785], device=self.device)
        else:
            self.u_hand_min = to_torch([-3.0], device=self.device)
            self.u_hand_max = to_torch([3.0], device=self.device)

        self._robot_effort_limits = to_torch(self._robot_effort_limits, device=self.device)

        # Define start pose for robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create table asset
        # table_pos = [0.5, 0.5, 0.3]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, *[0.8, 0.8, table_thickness], table_opts)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(.7, 0., 0.25)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table2_start_pose = gymapi.Transform()
        table2_start_pose.p = gymapi.Vec3(.0, -0.9, 0.25)
        table2_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        ## Load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.0001
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.convex_hull_downsampling = 30

        object_asset = self.gym.load_asset(self.sim, asset_root,
                                           "urdf/qb_hand/urdf/fresco.urdf",
                                           asset_options)

        self.object_default_state = torch.tensor([0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.7, 0.0, 0.03)
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = num_robot_bodies + num_table_bodies + num_table_bodies + num_object_bodies
        max_agg_shapes = num_robot_shapes + num_table_shapes + num_table_bodies + num_object_shapes

        self.goal_displacement = gymapi.Vec3(-0.2, 0.2, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        self.robots = []
        self.envs = []
        self.object_indices = []

        self.cameras = []
        self.camera_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True

        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

        if True:
            import open3d as o3d
            from bidexhands.utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else:
            self.pointCloudVisualizer = None

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robot actor and set properties
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            table_actor1 = self.gym.create_actor(env_ptr, table_asset, table2_start_pose, "table2", i, 2, 0)

            # Create object actor
            self._object_id = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 4, 0)
            fresco_color = gymapi.Vec3(0.8, 0.6, 0.2)
            self.gym.set_rigid_body_color(env_ptr, self._object_id, 0, gymapi.MESH_VISUAL_AND_COLLISION, fresco_color)

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs,
                                                0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            # self.goal_object_indices.append(goal_object_idx)

            camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.8, -0, 0.4),
                                         gymapi.Vec3(0.69, -0, 0.2))
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
            cam_vinv = torch.inverse(
                (torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),
                                    device=self.device)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            origin = self.gym.get_env_origin(env_ptr)
            self.env_origin[i][0] = origin.x
            self.env_origin[i][1] = origin.y
            self.env_origin[i][2] = origin.z

            self.camera_tensors.append(torch_cam_tensor)
            self.camera_view_matrixs.append(cam_vinv)
            self.camera_proj_matrixs.append(cam_proj)
            self.envs.append(env_ptr)
            self.cameras.append(camera_handle)

        # self.goal_states = self.object_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_init_state = self.goal_states.clone()

        # self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        # self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0

        self.handles = {
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                               robot_handle,
                                                               "qbhand_end_effector_link"),
            "fftip": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                           robot_handle,
                                                           "right_hand_v1_2_research_index_distal_link"),
            "thtip": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                           robot_handle,
                                                           "right_hand_v1_2_research_thumb_distal_link"),
            "object_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                                        self._object_id,
                                                                        "fresco"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        # get contact forces
        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, -1, 3)

        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._fftip_state = self._rigid_body_state[:, self.handles["fftip"], :]
        self._thtip_state = self._rigid_body_state[:, self.handles["thtip"], :]
        self._object_state = self._root_state[:, self._object_id, :]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr,
                                                         robot_handle)['qbhand_end_effector_fixed_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._vel_control = torch.zeros_like(self._pos_control)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7] if self.arm_control_type == "osc" else self._pos_control[:, :7]
        self._hand_control = self._pos_control[:, 7:]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

        self.down_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.grasp_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.target_pos = to_torch([0., -.9, 0.27], device=self.device).repeat((self.num_envs, 1))
        self.default_target_pos = torch.tensor([.0, -0.8, 0.27])

    def _update_states(self):
        self.states.update({
            # Robot
            "q_arm": self._q[:, :7],
            "q_hand": self._q[:, 7:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "fftip_pos": self._fftip_state[:, :3],
            "thtip_pos": self._thtip_state[:, :3],
            # Object
            "object_quat": self._object_state[:, 3:7],
            "object_pos": self._object_state[:, :3],
            "object_pos_relative": self._object_state[:, :3] - self._eef_state[:, :3],
            "object_fftip_pos_relative": self._object_state[:, :3] - self._fftip_state[:, :3],
            "object_thtip_pos_relative": self._object_state[:, :3] - self._thtip_state[:, :3],
            "target_pos_relative": self._object_state[:, :3] - self.target_pos,
            "target_pos": self.target_pos,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # update contact forces
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_robot_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        self._refresh()

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self._root_state[self.object_indices, 0:7]
        self.object_pos = self._root_state[self.object_indices, 0:3]
        self.object_rot = self._root_state[self.object_indices, 3:7]
        self.object_linvel = self._root_state[self.object_indices, 7:10]
        self.object_angvel = self._root_state[self.object_indices, 10:13]

        self.compute_full_state()

    def compute_full_state(self,):

        obs = ["object_pos", "object_quat", "object_pos_relative", "target_pos", "target_pos_relative", "eef_pos",
               "eef_quat"]
        self.obs_buf[:, 0:self.full_state] = torch.cat([self.states[ob] for ob in obs], dim=-1)

        goal_obs_start = self.full_state

        # plot an image every step of env 0
        self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
        camera_rgba_image = self.camera_visulization(is_depth_image=False)
        print(camera_rgba_image)
        plt.imshow(camera_rgba_image)
        plt.pause(1e-9)

        # ########################## After ##################
        # Convert camera tensors, view matrices, and projection matrices to device
        depth_buffer = [tensor.to(self.device) for tensor in self.camera_tensors]
        depth_buffer_tensor = torch.stack(depth_buffer, dim=0)
        vinv = [matrix.to(self.device) for matrix in self.camera_view_matrixs]
        vinv_tensor = torch.stack(vinv, dim=0)
        proj = [matrix.to(self.device) for matrix in self.camera_proj_matrixs]
        proj_tensor = torch.stack(proj, dim=0)

        fu = 2 / proj_tensor[:, 0, 0]
        fv = 2 / proj_tensor[:, 1, 1]

        width = self.camera_props.width
        height = self.camera_props.height
        centerU = width / 2
        centerV = height / 2
        Z = depth_buffer_tensor
        u = self.camera_u2
        v = self.camera_v2
        fu = fu.view(self.num_envs, 1, 1)  # Reshaping b to [2, 1, 1]
        fv = fv.view(self.num_envs, 1, 1)  # Reshaping b to [2, 1, 1]
        X = -(u - centerU) / width * Z * fu
        Y = (v - centerV) / height * Z * fv
        Z = Z.view(Z.shape[0], -1)
        valid = Z > -10

        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        ones = torch.ones(self.num_envs, X.shape[1], device=self.device)
        position = torch.stack((X, Y, Z, ones), dim=1)
        valid_expanded = valid.unsqueeze(1).repeat(1, 4, 1)  # Shape: [2, 4, 56000]
        # position = position[valid_expanded]  # Shape: [2, 4, 56000]       # removes False values
        position = torch.where(valid_expanded, position, torch.zeros_like(position))  # False values are 0.0
        position = position.view(self.num_envs, 4, -1)
        position = position.permute(0, 2, 1)
        position = position @ vinv_tensor
        points_envs = position[:, :, 0:3]

        sample_mathed = 'random'

        # remove points below 4 cm in z axis
        mask_if = points_envs[:, :, 2] > 0.277
        mask_if = mask_if.unsqueeze(-1).expand(-1, -1, 3)
        eff_points = torch.where(mask_if, points_envs, torch.zeros_like(points_envs) - 0.001)
        if sample_mathed == 'random':
            # sort in ascending order to remove the table points
            # norm_tensor = torch.norm(eff_points, p=2, dim=2)  # compute norm of x, y, z
            #
            # sorted_tensor, sorted_indices = torch.sort(norm_tensor, dim=1, descending=True)
            # sorted_indices_exp = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)  # expand tensor for 3D
            # sorted_tensor = eff_points.gather(1, sorted_indices_exp)

            row_total = int(points_envs.shape[1])  # select only 1/8 of tensor to not select table points
            sampled_points = points_envs[:, torch.randint(low=0, high=row_total, size=(self.pointCloudDownsampleNum,)), :]

        point_clouds = sampled_points

        if self.pointCloudVisualizer is not None:
            import open3d as o3d
            points = point_clouds[0, :, :3].cpu().numpy()
            # colors = plt.get_cmap()(point_clouds[0, :, 3].cpu().numpy())
            self.o3d_pc.points = o3d.utility.Vector3dVector(points)
            # self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])

        if not self.pointCloudVisualizerInitialized:
            self.pointCloudVisualizer.add_geometry(self.o3d_pc)
            self.pointCloudVisualizerInitialized = True
        else:
            self.pointCloudVisualizer.update(self.o3d_pc)

        self.gym.end_access_image_tensors(self.sim)
        # point cloud minus origin?? to normalize to 0
        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)

        point_clouds_start = goal_obs_start
        self.obs_buf[:, point_clouds_start:].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))

    def reset(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        pos = tensor_clamp(self.robot_default_dof_pos.unsqueeze(0),
                           self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits)

        # Reset object states by sampling random poses
        self._reset_init_object_state(env_ids=env_ids)
        self._object_state[env_ids] = self._init_object_state[env_ids]

        # # Reset target place location
        #self.target_pos = self.default_target_pos.repeat(len(env_ids), 1).to(device=self.device)
        #self.target_pos[:, :2] = self.target_pos[:, :2] + \
        #                         1.0 * self.start_position_noise * \
        #                         (torch.rand(len(env_ids), 2, device=self.device) - 0.5)

        # # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._vel_control[env_ids, :] = torch.zeros_like(pos)
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_velocity_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._vel_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update object states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        # self.lift_reward_list = []
        # self.lift_height_list = []
        # self.place_reward_list = []
        # self.dist_reward_list = []
        # self.fintip_reward_list = []

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # self.successes[env_ids] = 0

    def _reset_init_object_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = self.object_default_state.repeat(num_resets, 1).to(device=self.device)
        sampled_cube_state[:, :2] = sampled_cube_state[:, :2] + \
                                    1.0 * self.start_position_noise * \
                                    (torch.rand(num_resets, 2, device=self.device) - 0.5)

        # # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        self._init_object_state[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        # j_eef_inv = m_eef @ self._j_eef @ mm_inv
        # u_null = self.kd_null * -qd + self.kp_null * (
        #         (self.robot_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        # u_null[:, 7:] *= 0
        # u_null = self._mm @ u_null.unsqueeze(-1)
        # u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._robot_effort_limits[:7].unsqueeze(0), self._robot_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        if self.arm_control_type == "osc":
            u_arm, u_hand = self.actions[:, :6], self.actions[:, 6:]
            u_arm = u_arm * self.cmd_limit / self.action_scale
            u_arm = self._compute_osc_torques(dpose=u_arm)
        else:
            u_arm, u_hand = self.actions[:, :7], self.actions[:, 7:]
            u_arm = unscale_transform(u_arm,
                                      self.robot_dof_lower_limits[:7],
                                      self.robot_dof_upper_limits[:7])

            u_arm = tensor_clamp(u_arm,
                                 self.robot_dof_lower_limits[:7],
                                 self.robot_dof_upper_limits[:7])

        # u_hand = unscale_transform(u_hand, self.u_hand_min, self.u_hand_max)
        # u_hand = tensor_clamp(u_hand, self.u_hand_min, self.u_hand_max)

        # self._arm_control[:, :] = u_arm
        # Deploy actions
        if self.arm_control_type == "osc":
            self._effort_control[:, :7] = u_arm
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

        # Current pos and vel state of each DOF
        pos_tensor = self._dof_state[:, :, 0]
        vel_tensor = self._dof_state[:, :, 1]

        # == Hand control
        if self.hand_control_type == 'binary':
            self._hand_control[:, :] = u_hand
            # Fix knuckle joints
            self._hand_control[:, 0] = -0.1
            self._hand_control[:, 7] = 0.1
            self._hand_control[:, 14] = 0.0
            self._hand_control[:, 21] = 0.0
            self._hand_control[:, 28] = 1.5
            self._hand_control[:, 29] = 0.7
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # draw a sphere of where the place location is
        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                pos = gymapi.Vec3(self.target_pos[i][0], self.target_pos[i][1], self.target_pos[i][2])
                target_pos = gymapi.Transform(pos)
                gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], target_pos)

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0],
                                                                       gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                       to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Image.fromarray(camera_image)

        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0],
                                                                      gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Image.fromarray(camera_image)

        return camera_image


#####################################################################
# ##=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_robot_reward(
        rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
        max_episode_length: float,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # # Distance from the hand to the object
    # goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance
    #
    # # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2 * (dist_reward_scale))

    # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # # Fall penalty: distance to the goal is larger than a threashold
    # reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)
    #
    # # Check env termination conditions, including maximum success number
    # resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        # progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf),
        #                            progress_buf)
        resets = torch.where(successes >= max_consecutive_successes)
    # resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    # num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes)

    cons_successes = consecutive_successes

    return reward, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def random_pos(num: int, device: str) -> torch.Tensor:
    radius = 0.8
    height = 0.03
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.tensor([height], device=device).repeat((num, 1))

    return torch.cat((x[:, None], y[:, None], z), dim=-1)


@torch.jit.script
def remap(x: torch.Tensor, l1: float, h1: float, l2: float, h2: float) -> torch.Tensor:
    return l2 + (x - l1) * (h2 - l2) / (h1 - l1)


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


@torch.jit.script
def to_rads(x):
    return (x * 3.14159265359) / 180.
from typing import Any
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot

from .config_spot import SpotConfig

import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from scipy.spatial.transform import Rotation as R
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.api import arm_command_pb2
from bosdyn.util import seconds_to_duration


import numpy as np
import time
import cv2

class Spot(Robot):
    config_class = SpotConfig
    name = "Spot"
    def __init__(self, config: SpotConfig):
        super().__init__(config)
        # save config
        self.config = config
        # create varables
        bosdyn.client.util.setup_logging()
        self._sdk = bosdyn.client.create_standard_sdk('Spot_LeRobot_Robot')
        self.robot = None      
        self.robot_state_client: RobotStateClient
        self.robot_command_client: RobotCommandClient
        self.lease_client: LeaseClient
        self.image_client: ImageClient
        self.robot_state = None
        
        self._lease_keepalive = None
        self._is_connected = False

        self._VELOCITY_CMD_DURATION = 0.1
        self._VELOCITY_CMD_DURATION_ARM = 0.1

        # Inside __init__
        self.last_arm_pose = {
            "x": 0.3, "y": 0.0, "z": 0.2,
            "rot.w": 0.0,  "rot.x": 0.0, "rot.y": 0.0, "rot.z": 0.0
        }
        self.carry_mode = True
        self.carry_flipped = False
    
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot = self._sdk.create_robot(self.config.robot_ip)
        self.robot.authenticate(self.config.robot_user,
                                  self.config.robot_password)
        bosdyn.client.util.authenticate(self.robot)

        self.robot.time_sync.wait_for_sync()
        
        assert not self.robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'        

        self.robot_state_client     = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.robot_command_client   = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client           = self.robot.ensure_client(LeaseClient.default_service_name)
        self.image_client           = self.robot.ensure_client(ImageClient.default_service_name)
        self.robot_state = self.robot_state_client.get_robot_state()

        self._lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True
        )
        
        self._is_connected = True

   
        # blocking call
        self.robot.logger.info('Powering on robot... This may take several seconds.')
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'
        self.robot.logger.info('Robot powered on.')
        # stand up the robot
        self.robot.logger.info('Commanding robot to stand...')
        blocking_stand(self.robot_command_client, timeout_sec=10)
        self.robot.logger.info('Robot standing.')

        self.configure()

    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = {}

        state = self.robot_state_client.get_robot_state()
        image_responses = self.image_client.get_image_from_sources(
            ['hand_color_image']
            )
                
        joint_states = state.kinematic_state.joint_states

        # Extract Arm Joints (sh0, sh1, el0, el1, wr0, wr1)
        arm_joints = sorted(
            [j for j in joint_states if j.name.startswith("arm0") and not j.name.endswith("f1x")],
            key=lambda x: x.name
        )
        arm_joint_pos = np.array([j.position.value for j in arm_joints], dtype=np.float32)
        arm_joint_vel = np.array([j.velocity.value for j in arm_joints], dtype=np.float32)
        arm_joint_load = np.array([j.load.value for j in arm_joints], dtype=np.float32)
        for i, angle in enumerate(arm_joint_pos):
            obs_dict[f"arm.joint{i+1}.pos"]   = angle
            obs_dict[f"arm.joint{i+1}.vel"]   = arm_joint_vel[i]
            obs_dict[f"arm.joint{i+1}.load"]  = arm_joint_load[i]

        
        # Extract Gripper Joint (arm0.f1x)
        gripper_joint = next((j for j in joint_states if j.name == "arm0.f1x"), None)
        manipulator_state = state.manipulator_state     
        obs_dict[f"gripper.pos"] = manipulator_state.gripper_open_percentage / 100.0
        obs_dict[f"gripper.vel"] = gripper_joint.velocity.value
        obs_dict[f"gripper.load"] = gripper_joint.load.value
    
        # Arm Pose
        body_tform_hand = get_a_tform_b(
            state.kinematic_state.transforms_snapshot,
            GRAV_ALIGNED_BODY_FRAME_NAME,
            HAND_FRAME_NAME
        )
        hand_pose = np.array([
            body_tform_hand.x, body_tform_hand.y, body_tform_hand.z,
            body_tform_hand.rot.w, body_tform_hand.rot.x, body_tform_hand.rot.y, body_tform_hand.rot.z
        ], dtype=np.float32)
        obs_dict["arm.x"] = body_tform_hand.x
        obs_dict["arm.y"] = body_tform_hand.y
        obs_dict["arm.z"] = body_tform_hand.z
        obs_dict["arm.rot.w"] = body_tform_hand.rot.w
        obs_dict["arm.rot.x"] = body_tform_hand.rot.x
        obs_dict["arm.rot.y"] = body_tform_hand.rot.y
        obs_dict["arm.rot.z"] = body_tform_hand.rot.z        
        # Base
        base_vel_linear = state.kinematic_state.velocity_of_body_in_vision.linear
        base_vel_angular = state.kinematic_state.velocity_of_body_in_vision.angular
        obs_dict["base.x.vel"] = base_vel_linear.x
        obs_dict["base.y.vel"] = base_vel_linear.y
        obs_dict["base.rot.vel"] = base_vel_angular.z

        # images
        for resp in image_responses:
            if resp.source.name == 'hand_color_image':
                dtype = np.uint8
                img = cv2.imdecode(np.frombuffer(resp.shot.image.data, dtype=dtype), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                obs_dict = {
                    "images.gripper_cam" : rgb_img,
                }

        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        v_x = action.get("base.x.vel", 0.0)
        v_y = action.get("base.y.vel", 0.0)
        v_rot = action.get("base.rot.vel", 0.0)

        base_command = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x, v_y=v_y, v_rot=v_rot
        )

        # handle activation/deactivation off carry mode
        if action.get("arm_carry_enabled") and not self.carry_flipped:
            self.carry_mode = not self.carry_mode
            self.carry_flipped = True
        elif not action.get("arm_carry_enabled") and self.carry_flipped:
            self.carry_flipped = False
        
        if action.get("arm_control"):
            # Update last known pose
            self.last_arm_pose["x"] = action.get("arm.x", self.last_arm_pose["x"])
            self.last_arm_pose["y"] = action.get("arm.y", self.last_arm_pose["y"])
            self.last_arm_pose["z"] = action.get("arm.z", self.last_arm_pose["z"])
            self.last_arm_pose["rot.w"] = action.get("arm.rot.w", self.last_arm_pose["rot.w"])
            self.last_arm_pose["rot.x"] = action.get("arm.rot.x", self.last_arm_pose["rot.x"])
            self.last_arm_pose["rot.y"] = action.get("arm.rot.y", self.last_arm_pose["rot.y"])
            self.last_arm_pose["rot.z"] = action.get("arm.rot.z", self.last_arm_pose["rot.z"])
            
            arm_command = RobotCommandBuilder.arm_pose_command(
                self.last_arm_pose["x"], self.last_arm_pose["y"], self.last_arm_pose["z"],
                self.last_arm_pose["rot.w"], 
                self.last_arm_pose["rot.x"], 
                self.last_arm_pose["rot.y"], 
                self.last_arm_pose["rot.z"], 
                GRAV_ALIGNED_BODY_FRAME_NAME,
                self._VELOCITY_CMD_DURATION_ARM
            )
        elif self.carry_mode:
            arm_command = RobotCommandBuilder.arm_carry_command()
        else:           
            arm_command = RobotCommandBuilder.arm_pose_command(
                self.last_arm_pose["x"], self.last_arm_pose["y"], self.last_arm_pose["z"],
                self.last_arm_pose["rot.w"], 
                self.last_arm_pose["rot.x"], 
                self.last_arm_pose["rot.y"], 
                self.last_arm_pose["rot.z"],  
                GRAV_ALIGNED_BODY_FRAME_NAME,
                self._VELOCITY_CMD_DURATION_ARM
            )


        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            action.get("gripper.pos", 0.0)
        )

        command = RobotCommandBuilder.build_synchro_command(
            base_command, arm_command, gripper_command
        )

        # 5. Send to Robot
        end_time_secs = time.time() + self._VELOCITY_CMD_DURATION
        self.robot_command_client.robot_command_async(
            command=command,
            end_time_secs=end_time_secs
        )

        return action


    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")
    
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
            self._lease_keepalive = None
        assert not self.robot.is_powered_on(), 'Robot power off failed.'
        self.robot.logger.info('Robot safely powered off.')
        self._is_connected = False
        self.robot.logger.info('Robot safely disconnected.')


    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "base.x.vel": float,
            "base.x.vel": float,
            "base.rot.vel": float,
            "arm.joint1.pos": float,
            "arm.joint2.pos": float,
            "arm.joint3.pos": float,
            "arm.joint4.pos": float,
            "arm.joint5.pos": float,
            "arm.joint6.pos": float,           
            "arm.x": float,
            "arm.y": float,
            "arm.z": float,
            "arm.rot.w": float,
            "arm.rot.x": float,
            "arm.rot.y": float,
            "arm.rot.z": float,
            "gripper.pos": float,            
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            #cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict:
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:        
        return self._is_connected # and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass
    
    def configure(self) -> None:
        return
    


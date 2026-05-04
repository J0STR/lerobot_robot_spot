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
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.api import arm_command_pb2
from bosdyn.util import seconds_to_duration


import numpy as np
import time

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
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0 # yaw
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
        
        self.robot_state = self.robot_state_client.get_robot_state()
        image_responses = self.image_client.get_image_from_sources(
            ['hand_color_image']
            )

        obs_dict = {}

        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        v_x = action.get("x_axis.vel", 0.0)
        v_y = action.get("y_axis.vel", 0.0)
        v_rot = action.get("rotation.vel", 0.0)

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
            self.last_arm_pose["roll"] = action.get("arm.roll", self.last_arm_pose["roll"])
            self.last_arm_pose["pitch"] = action.get("arm.pitch", self.last_arm_pose["pitch"])
            self.last_arm_pose["yaw"] = action.get("arm.yaw", self.last_arm_pose["yaw"])

            rpy = [self.last_arm_pose["roll"], self.last_arm_pose["pitch"], self.last_arm_pose["yaw"]]
            quat = R.from_euler('xyz', rpy).as_quat()
            
            arm_command = RobotCommandBuilder.arm_pose_command(
                self.last_arm_pose["x"], self.last_arm_pose["y"], self.last_arm_pose["z"],
                quat[3], quat[0], quat[1], quat[2], 
                GRAV_ALIGNED_BODY_FRAME_NAME,
                self._VELOCITY_CMD_DURATION_ARM
            )
        elif self.carry_mode:
            arm_command = RobotCommandBuilder.arm_carry_command()
        else:
            rpy = [self.last_arm_pose["roll"], self.last_arm_pose["pitch"], self.last_arm_pose["yaw"]]
            quat = R.from_euler('xyz', rpy).as_quat()
            
            arm_command = RobotCommandBuilder.arm_pose_command(
                self.last_arm_pose["x"], self.last_arm_pose["y"], self.last_arm_pose["z"],
                quat[3], quat[0], quat[1], quat[2], 
                GRAV_ALIGNED_BODY_FRAME_NAME,
                self._VELOCITY_CMD_DURATION_ARM
            )


        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            action.get("gripper.action", 0.0)
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

        self.carry_last_state = action.get("arm_carry_enabled")

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
            "x_axis.vel": float,
            "y_axis.vel": float,
            "rotation.vel": float,
            "arm.x": float,
            "arm.y": float,
            "arm.z": float,
            "arm.roll": float,
            "arm.pitch": float,
            "arm.yaw": float,
            "gripper.action": float,
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
    


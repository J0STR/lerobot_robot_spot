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
        self.robot_state = None
        
        self._lease_keepalive = None
        self._is_connected = False

        self._VELOCITY_CMD_DURATION = 0.1
        self._VELOCITY_CMD_DURATION_ARM = 0.5
    
    
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

        obs_dict = {}

        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        v_x = action.get("x_axis.vel", 0.0)
        v_y = action.get("y_axis.vel", 0.0)
        v_rot = action.get("rotation.vel", 0.0)
        
        base_command = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x, v_y=v_y, v_rot=v_rot
        )

        rpy = [action.get("arm.roll", 0.0), action.get("arm.pitch", 0.0), action.get("arm.yaw", 0.0)]
        quat = R.from_euler('xyz', rpy).as_quat() # Returns [x, y, z, w]
        
        if action.get("arm_control"):
            arm_command = RobotCommandBuilder.arm_pose_command(
                action.get("arm.x", 0.3), # Default reach
                action.get("arm.y", 0.0), 
                action.get("arm.z", 0.0),
                quat[3], quat[0], quat[1], quat[2], # Spot wants [w, x, y, z]
                GRAV_ALIGNED_BODY_FRAME_NAME,
                self._VELOCITY_CMD_DURATION_ARM
            )

            gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
                action.get("gripper.action", 0.0)
            )

            command = RobotCommandBuilder.build_synchro_command(
                base_command, arm_command, gripper_command
            )
        else:
            gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
                action.get("gripper.action", 0.0)
            )      
            command = RobotCommandBuilder.build_synchro_command(base_command,
                                                                           gripper_command)

        # 5. Send to Robot
        end_time_secs = time.time() + self._VELOCITY_CMD_DURATION
        self.robot_command_client.robot_command(
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
    


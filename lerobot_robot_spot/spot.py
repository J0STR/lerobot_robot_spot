from typing import Any
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot

from .config_spot import SpotConfig

import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import trajectory_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration

import numpy as np
import time

class Dual_xArm7(Robot):
    config_class = SpotConfig
    name = "Spot"
    def __init__(self, config: SpotConfig):
        super().__init__(config)
        # save config
        self._config = config
        # create varables
        bosdyn.client.util.setup_logging()
        self._sdk = bosdyn.client.create_standard_sdk('Spot_LeRobot_Robot')
        self._robot = None      
        self.robot_state_client: RobotStateClient
        self.robot_command_client: RobotCommandClient
        self.lease_client: LeaseClient
        self.robot_state = None
        
        
        self._is_connected = False

        self._VELOCITY_CMD_DURATION = 0.1
    
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._robot = self._sdk.create_robot(self._config.robot_ip)
        self._robot.authenticate(self._config.robot_user,
                                  self._config.robot_password)
        bosdyn.client.util.authenticate(self._robot)

        self._robot.time_sync.wait_for_sync()
        
        assert not self._robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'        

        self.robot_state_client     = self._robot.ensure_client(RobotStateClient.default_service_name)
        self.robot_command_client   = self._robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client           = self._robot.ensure_client(LeaseClient.default_service_name)
        self.robot_state = self.robot_state_client.get_robot_state()
        
        self._is_connected = True

        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
            # blocking call
            self._robot.logger.info('Powering on robot... This may take several seconds.')
            self._robot.power_on(timeout_sec=20)
            assert self._robot.is_powered_on(), 'Robot power on failed.'
            self._robot.logger.info('Robot powered on.')

        self.configure()

    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        
        self.robot_state = self.robot_state_client.get_robot_state()

        obs_dict = {}
        
        print(self.robot_state)

        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_vel = {key.removesuffix(".vel"): val for key, val in action.items()}
        action = goal_vel 

        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
            if goal_vel['x_axis'] != 0 or goal_vel['y_axis'] !=0 or goal_vel['rot_axis'] !=0:
                    command = RobotCommandBuilder.synchro_velocity_command(v_x=goal_vel['x_axis'],
                                                                            v_y=goal_vel['y_axis'],
                                                                            v_rot=goal_vel['rotation'])
                    end_time_secs=time.time() + self._VELOCITY_CMD_DURATION
                    self.robot_command_client.robot_command(command=command,
                                                            end_time_secs=end_time_secs) 
    

        return action


    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")
        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
            self._robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not self._robot.is_powered_on(), 'Robot power off failed.'
            self._robot.logger.info('Robot safely powered off.')


    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "x_axis.vel": float,
            "y_axis.vel": float,
            "rotation.vel": float,
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
    


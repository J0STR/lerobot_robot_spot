from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras import ColorMode, Cv2Rotation
from lerobot.robots import RobotConfig

from login_data import * # add your own info here


@RobotConfig.register_subclass("Spot")
@dataclass
class SpotConfig(RobotConfig):
    id: str = 'Spot_with_Arm'
    robot_ip: str = "192.168.80.3"
    robot_user: str = user_name
    robot_password: str = user_password


    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top_down": RealSenseCameraConfig(
                serial_number_or_name="352122273665",
                fps=30,
                width=640,
                height=480),
            # "wurm_eye": RealSenseCameraConfig(
            #     serial_number_or_name="409122274688",
            #     fps=30,
            #     width=640,
            #     height=480),
            "wrist_right": RealSenseCameraConfig(
                serial_number_or_name="352122273091",
                fps=30,
                width=640,
                height=480),
            "wrist_left": RealSenseCameraConfig(
                serial_number_or_name="409122271056",
                fps=30,
                width=640,
                height=480),
                
        }
    )
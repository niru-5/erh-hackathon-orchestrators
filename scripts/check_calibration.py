"""Read-only check that the saved SO-101 calibration files match what is stored in the motors.

Only reads from the motor buses: no torque is enabled and nothing is written or moved.

Usage: .venv/bin/python scripts/check_calibration.py
"""

import os

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115116-if00")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116682-if00")
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "so101")
LEADER_ID = os.environ.get("LEADER_ID", "so101")

for name, dev in [
    ("follower", SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))),
    ("leader", SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id=LEADER_ID))),
]:
    print(f"{name}: file {dev.calibration_fpath} loaded={bool(dev.calibration)}")
    dev.bus.connect()  # handshake only: no torque, no writes
    try:
        motor_cal = dev.bus.read_calibration()
        diffs = {m: (vars(dev.calibration[m]), vars(c)) for m, c in motor_cal.items() if vars(c) != vars(dev.calibration[m])}
        print(f"{name}: motors match file = {dev.bus.is_calibrated}", diffs or "")
        print(f"{name}: raw positions = {dev.bus.sync_read('Present_Position', normalize=False)}")
    finally:
        dev.bus.disconnect(disable_torque=False)

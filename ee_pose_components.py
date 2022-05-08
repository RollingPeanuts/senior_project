# This script makes the end-effector go to a specific pose by defining the pose components
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=px100'
# Then change to this directory and type 'python ee_pose_components.py'
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from collections import namedtuple
from typing import Tuple
import math

# 3D coordinates and orientation for robot arm to reach
Robot_Coordinate = namedtuple('Robot_Coordinate', ['x', 'y', 'z', 'pitch'])

# 2D coordinate representing the position on the board with the given row and column
Board_Coordinate = namedtuple('Board_Coordinate', ['row', 'col'])

# Arbitrary coordinate to calibrate rest of positions of board
# Choosing coordinates closer to the center of the board will yield more accurate robot arm positions
# base coordinates represent best guess of coordinate location, to be calibrated
# ADJUST AS NEEDED
base_index = Board_Coordinate(-6, 0)
base_coordinates = Robot_Coordinate(0.195, -0.01, 0.05, 1.5) 

# Length of the side of a hexagon of the board (m)
unit_length = 0.045


# Run manual calibration on configured base coordinate. Base coordinate will be used to extrapolate all remaining 
def calibrate_base_coordinates(bot: InterbotixManipulatorXS) -> None:
    global base_coordinates
    bot.arm.go_to_home_pose()
    bot.arm.set_ee_pose_components(x=base_coordinates.x, y=base_coordinates.y, z=base_coordinates.z, pitch=base_coordinates.pitch)

    # Run calibration until user is satisfied with precision
    print("Calibrating for point on row {}, col {}".format(base_index.row, base_index.col))
    print("Provide requests in the form of \"[x/y/z] [movement in cm]\", or \"done\" to finish calibration")
    print(" e.g. \"x -4\" moves the arm 4 cm backwards")
    while (True):
        new_coordinates, calibration_is_satisfied = _handle_calibration_request()
        if (calibration_is_satisfied):
            break

        # Only store the new coordinates if such a move is possible for the arm 
        _, is_valid_move = bot.arm.set_ee_pose_components(x=new_coordinates.x, y=new_coordinates.y, z=new_coordinates.z, pitch=new_coordinates.pitch)
        if (is_valid_move):
            base_coordinates = new_coordinates


# Process calibration request from user and update base coordinates to reflect requested change
# Returns whether calibration has finished or not
def _handle_calibration_request() -> Tuple[Robot_Coordinate, bool]:
    while (True):
        calibration_request = input().split()
        request_length = len(calibration_request)
        
        if (request_length == 1 and calibration_request[0] == "done"):
            return (None, True)
        elif (request_length == 2):
            if (calibration_request[0] == "x"):
                return (Robot_Coordinate(base_coordinates.x + float(calibration_request[1]) / 100.0, base_coordinates.y, base_coordinates.z, base_coordinates.pitch), False)
            elif (calibration_request[0] == "y"):
                return (Robot_Coordinate(base_coordinates.x, base_coordinates.y + float(calibration_request[1]) / 100.0, base_coordinates.z, base_coordinates.pitch), False)
            elif (calibration_request[0] == "z"):
                return (Robot_Coordinate(base_coordinates.x, base_coordinates.y, base_coordinates.z + float(calibration_request[1]) / 100.0, base_coordinates.pitch), False)
        print("Invalid Request")


# Find the coordinate of a position based on the calibrated base coordinate
def get_destination_coordinates(row: int, col: int) -> Robot_Coordinate:
    # rows are uniformally distributed s.t. every two rows are seperated by sqrt(3) / 2 * the unit length of a side of the hexagon
    x_coord = (row - base_index.row) * (unit_length * math.sqrt(3) / 4) + base_coordinates.x

    # cols are semi uniformally distributed, every 4 rows are seperated by 1.5 * unit length, but there is deviation within the 4 rows
    col_offsets = [0, 0.0225, 0.03375, 0.045]
    base_y_offset = math.floor(base_index.col / 4) * (unit_length * 1.5) + col_offsets[base_index.col % 4]
    new_y_offset = math.floor(col / 4) * (unit_length * 1.5) + col_offsets[col % 4]
    y_coord = new_y_offset - base_y_offset + base_coordinates.y

    return Robot_Coordinate(x_coord, y_coord, base_coordinates.z, base_coordinates.pitch)


# Grabs the specified type of piece
# TODO: Create 3 places to grab pieces
def grab_obj(bot: InterbotixManipulatorXS, piece: int) -> None:
    bot.arm.go_to_home_pose()
    bot.gripper.open()
    bot.arm.set_ee_pose_components(x=0, y=0.13, z=0.22, pitch=0.2)
    bot.arm.set_ee_pose_components(x=0, y=0.13, z=0.027, pitch=1.5)
    bot.gripper.close()
    #bot.arm.set_ee_pose_components(x=0.15, y=0.15, z=0.22, pitch=0.2)
    bot.arm.set_ee_pose_components(x=0, y=0.13, z=0.22, pitch=0.2)
    bot.arm.go_to_home_pose()


# Places an object at set position
def place_obj(bot: InterbotixManipulatorXS, coords: Robot_Coordinate) -> None:
    bot.arm.go_to_home_pose()
    bot.arm.set_ee_pose_components(x=coords.x, y=coords.y, z=0.22, pitch=0.2) # position arm to hover over drop position
    #bot.arm.set_ee_pose_components(x=0.15, y=0.1, z=0.035, pitch=1.5) # most left, 2nd farthest out
    #bot.arm.set_ee_pose_components(x=0.15, y=0.045, z=0.035, pitch=1.5) # second farthest left, 2nd farthest out
    #bot.arm.set_ee_pose_components(x=0.15, y=-0.045, z=0.035, pitch=1.5) # second farthest right, 2nd farthest out
    #bot.arm.set_ee_pose_components(x=0.15, y=-0.1, z=0.035, pitch=1.5) # farthest right, 2nd farthest out
    bot.arm.set_ee_pose_components(x=coords.x, y=coords.y, z=coords.z, pitch=1.5) # middle, farthest out
    #bot.arm.set_ee_pose_components(x=0.105, y=0, z=0.035, pitch=1.5) # middle, closest 
    #bot.arm.set_ee_pose_components(x=0.205, y=0.025, z=0.05, pitch=1.1) # middle, closest
    bot.gripper.open()
    bot.arm.set_ee_pose_components(x=coords.x, y=coords.y, z=0.22, pitch=0.2)
    bot.arm.go_to_home_pose()
    bot.gripper.close()


def main():
    bot = InterbotixManipulatorXS("px100", "arm", "gripper")
    bot.arm.go_to_sleep_pose()

    calibrate_base_coordinates(bot)

    #TODO: Finalize conversion from hexagon number to row/col format
    while (True):
        (row, col, piece) = input("Insert move of robot (row, col, piece):").split()
        coords = get_destination_coordinates(int(row), int(col))

        #bot.arm.go_to_home_pose()
        #grab_obj(bot, piece)
        place_obj(bot, coords)
        bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()

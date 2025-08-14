import rospy
import numpy as np
import os
from std_msgs.msg import Bool
from dynamixel_sdk import *  # 使用Dynamixel SDK
import numpy as np
import cv2

class GripperStateListener:
    def __init__(self):
        self.gripper_state = False
        rospy.Subscriber('/right_gripper_topic', Bool, self.callback)

    def callback(self, msg):
        self.gripper_state = msg.data

    def get_state(self):
        return self.gripper_state
    
class Gripper_Controller:
    def __init__(self, portHandler, packetHandler):
        self.gripper_state = 2800
        self.portHandler = portHandler
        self.packetHandler = packetHandler

    def ctrl_gripper(self, goal_position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_position)
        while True:
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID, ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            if not abs(goal_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
                break
        self.gripper_state = dxl_present_position

    # def control_thread(self, command):
    #     print(f"Received command: {command}")
    #     DELTA_GRIPPER_CMD = 50
    #     if command:
    #         target_action = 1500
    #     else:
    #         target_action = 2800

    #     gripper_state = self.gripper_state
    #     gripper_action = np.clip(target_action, gripper_state - DELTA_GRIPPER_CMD, gripper_state + DELTA_GRIPPER_CMD)
    #     self.ctrl_gripper(int(gripper_action))
    def control_thread(self, command):
        print(f"Received command: {command}")
        DELTA_GRIPPER_CMD = 100

        if command:
            target_action = 2100
        else:
            target_action = 2800
        gripper_state = self.gripper_state
        gripper_action = np.clip(target_action, gripper_state - DELTA_GRIPPER_CMD, gripper_state + DELTA_GRIPPER_CMD)
        self.ctrl_gripper(int(gripper_action))


if __name__ == '__main__':
    
    if os.name == 'nt':
        import msvcrt
        def getch():
            return msvcrt.getch().decode()
    else:
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        def getch():
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

    PROTOCOL_VERSION = 2.0
    DXL_ID = 7
    DEVICENAME = "/dev/ttyUSB1"
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    DXL_MOVING_STATUS_THRESHOLD = 200

    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    MY_DXL = 'X_SERIES'
    if MY_DXL == 'X_SERIES' or MY_DXL == 'MX_SERIES':
        ADDR_TORQUE_ENABLE = 64
        ADDR_GOAL_POSITION = 116
        ADDR_PRESENT_POSITION = 132
        DXL_MINIMUM_POSITION_VALUE = 0
        DXL_MAXIMUM_POSITION_VALUE = 4095
        BAUDRATE = 1000000

    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()

    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel has been successfully connected")
    rospy.init_node('right_gripper_sub_node')
    listener = GripperStateListener()
    gripper_controller = Gripper_Controller(portHandler, packetHandler)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        state = listener.get_state()
        gripper_controller.control_thread(state)
        rate.sleep()
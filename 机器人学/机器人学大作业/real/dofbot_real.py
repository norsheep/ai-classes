"""
Real Env
"""
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState

class RealEnv:
    def __init__(self,init_node=True):
        if init_node:
            rospy.init_node("Arm2")
        self.state = None
        self.pub = rospy.Publisher("/dofbot/cmd", JointState, queue_size=10)
        rospy.Subscriber("/dofbot/joint_state", JointState, self.callback)

    def callback(self, data: JointState):
        """设置state，在广播信息的时候也会被调用

        Args:
            data (JointState): 自身状态
        """
        self.state = np.asarray(data.position)

    def reset(self):
        """重置所有的信息为([90., 90., 90., 90., 90.], 10., 2000)
        """
        while self.state is None:
            time.sleep(0.1)
        self.send([90., 90., 90., 90., 90.], 10., 2000)
        time.sleep(2)

    def get_state(self):
        """返回各关节角度 + 夹爪 + time"""
        return self.state.copy()

    def send(self, angles, gripper, t):
        """广播一大堆信息

        Args:
            angles (maybe list[float]): 各个关节的角度
            gripper (maybe float): 夹爪角度，负值夹紧，正值张开
            t (maybe int): 应该是时间， 且以毫秒为单位
        """
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = np.round(angles).tolist() + [gripper] + [int(t)]
        msg.name = [f"Joint{i}" for i in range(6)]

        self.pub.publish(msg)

    def control_gripper(self, g):
        """控制夹爪角度g

        Args:
            g (maybe float): 夹爪角度，负值夹紧，正值张开
        """
        self.send(self.state[:-1], g, 100)
        time.sleep(0.1)

    def control_joints(self, p):
        """控制关节角度p, atol = 3

        Args:
            p (maybe list[float]): 一系列关节角度
        """
        while not rospy.is_shutdown():
            self.send(p, self.state[-1], 150)
            time.sleep(0.15)
            if np.isclose(self.state[:-1], p,atol=3.).all():
                break

    def step(self, joint=None, gripper=None):
        """传入至少一个机械臂参数(关节、夹爪),然后控制

        Args:
            joint (maybe list[float], optional): 一系列关节角度. Defaults to None.
            gripper (_type_, optional): 夹爪角度，负值夹紧，正值张开. Defaults to None.
        """
        if joint is not None:
            self.control_joints(joint)
        if gripper is not None:
            self.control_gripper(gripper)

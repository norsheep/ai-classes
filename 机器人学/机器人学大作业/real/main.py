import time
import numpy as np
import rospy
from dofbot_real import RealEnv

if __name__ == '__main__':
    env = RealEnv()
    env.reset()

    #points代表了一系列移动时的关节状态
    points = [
        np.asarray([90., 90., 90., 90., 90.]),  # 原始状态
        np.asarray([136.0, 50.0, 53.0, 1.0, 86.0]),  #移动到物块上方
        np.asarray([136.0, 50.0, 53.0, 1.0, 86.0]),  #夹住
        np.asarray([136.0, 70.0, 53.0, 1.0, 86.0]),  #抬起来
        np.asarray([180 - 138.0, 70.0, 53.0, 1.0, 86.0]),  #转动
        np.asarray([180 - 138.0, 55.0, 53.0, 1.0, 86.0]),  #放下
        # np.asarray([136.0, 90.0, 53.0, 1.0, 86.0]),
        # np.asarray([90.0, 50.0, 53.0, 1.0, 86.0]),
        np.asarray([180 - 138.0, 65.0, 53.0, 1.0, 86.0]),  #松开
    ]

    #每段路径的插值点数; spilt[i] 插值 point[i] - point[i+1]
    split = [30, 5, 30, 40, 10, 10]
    #夹爪的角度，负值表示和上一个相同; gripper[i]对应point[i]
    gripper = [-1, -1, 140., -1., -1, -1, 10.]

    def linear_interpolation(src, tat, n=10):
        """src - tat 插值n个点"""
        path = np.linspace(src, tat, num=n)
        return path

    for i in range(len(points) - 1):
        #下一阶段如果不需要夹爪，则仅移动机械臂；否则仅移动夹爪
        if gripper[i + 1] < 0.:
            path = linear_interpolation(points[i], points[i + 1], n=split[i])
        else:
            path = linear_interpolation(env.get_state()[-1],
                                        gripper[i + 1],
                                        n=split[i])

        #按照插值点控制路径
        for p in path:
            if gripper[i + 1] < 0.:
                env.step(p)
            else:
                env.step(gripper=p)

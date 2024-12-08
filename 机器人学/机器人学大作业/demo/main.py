from dofbot import DofbotEnv
import numpy as np
import time
import copy
from scipy.spatial.transform import Rotation as R
import time
import math

if __name__ == '__main__':
    env = DofbotEnv()
    env.reset()  # 重置环境
    Reward = False  # 是否完成任务
    '''
    constants here
    '''
    GRIPPER_DEFAULT_ANGLE = 20. / 180. * 3.1415  # 夹爪默认角度
    GRIPPER_CLOSE_ANGLE = -20. / 180. * 3.1415  # 夹爪闭合角度

    # define state machine
    INITIAL_STATE = 0  # 在初始位置
    GRASP_STATE = 1  # 抓取
    LIFT_STATE = 2  # 抬起
    PUT_STATE = 3  # 放置
    MOVE_STATE = 4  # 移动
    BACK_STATE = 5  # 返回
    current_state = INITIAL_STATE  # 当前状态

    initial_jointposes = [1.57, 0., 1.57, 1.57, 1.57]  # 初始关节角度

    # offset to grasp object 可以调整，能夹起来就行
    obj_offset = [-0.023, -0.023, 0.09]
    obj_offset2 = [-0.032, 0.032, 0.13]
    obj_offset3 = [-0.025, 0.025, 0.09]
    # 夹爪伸进来才能夹住  注意有42mm的偏差

    block_pos, block_orn = env.get_block_pose()  # 获取物块位姿xyz，四元数
    # print(block_pos, block_orn) # (0.2, 0.1, 0.014990199999999999)
    # 暂停
    # input()
    start_time = None

    while not Reward:
        '''
        获取物块位姿、目标位置和机械臂位姿，计算机器臂关节和夹爪角度
        使得机械臂夹取绿色物块，放置到紫色区域。
        code here
        '''

        if current_state == INITIAL_STATE:
            # target_pos储存目标位置,在物块位置的基础上增加一个偏差，抓夹的偏差已包含在DH建系中
            target_pos = (block_pos[0] + obj_offset[0],
                          block_pos[1] + obj_offset[1],
                          block_pos[2] + obj_offset[2])
            # 逆运动学得到目标关节角度，由物块决定抓夹角度
            target_state = env.dofbot_setInverseKine(target_pos,
                                                     -1 * block_orn)
            env.dofbot_control(target_state, GRIPPER_DEFAULT_ANGLE)

            # 检查是否到达目标位置一定范围内
            current_pos, _ = env.get_dofbot_jointPoses()
            if np.all(
                    np.isclose(np.array(current_pos),
                               np.array(target_state),
                               atol=1e-2)):
                current_state = GRASP_STATE
                start_time = time.time()

        elif current_state == GRASP_STATE:
            env.dofbot_control(target_state, GRIPPER_CLOSE_ANGLE)
            current_time = time.time()
            if (current_time - start_time > 2.0):
                current_state = LIFT_STATE
                # lift_pos储存抬起的目标位置，补充物块偏差,减少重复计算
                lift_pos = (block_pos[0] + obj_offset[0],
                            block_pos[1] + obj_offset[1],
                            block_pos[2] + obj_offset[2] + 0.02)

        elif current_state == LIFT_STATE:
            # 网上抬0.4
            lift_state = env.dofbot_setInverseKine(lift_pos, -1 * block_orn)
            env.dofbot_control(lift_state, GRIPPER_CLOSE_ANGLE)
            current_pos, _ = env.get_dofbot_jointPoses()

            # 检查是否到达目标位置一定范围内
            if (np.all(
                    np.isclose(np.array(current_pos),
                               np.array(lift_state),
                               atol=1e-2))):
                current_state = MOVE_STATE
                # move_pos储存移动的目标位置，补充物块偏差，减少重复计算
                target_pos = env.get_target_pose()
                move_pos = (target_pos[0] + obj_offset2[0],
                            target_pos[1] + obj_offset2[1],
                            target_pos[2] + obj_offset2[2])
                # self.target_pos = np.array([0.2, -0.1, 0.015])  原为0.15，感觉不大对，自行修改了一下
        elif current_state == MOVE_STATE:

            move_state = env.dofbot_setInverseKine(move_pos, -1 * block_orn)
            env.dofbot_control(move_state, GRIPPER_CLOSE_ANGLE)
            # 检查是否到达目标位置一定范围内
            current_pos, _ = env.get_dofbot_jointPoses()
            if (np.all(
                    np.isclose(np.array(current_pos),
                               np.array(move_state),
                               atol=1e-2))):
                current_state = BACK_STATE
                #节省重复计算
                target_pos = env.get_target_pose()
                final_pos = (target_pos[0] + obj_offset3[0],
                             target_pos[1] + obj_offset3[1],
                             target_pos[2] + obj_offset3[2])

        elif current_state == BACK_STATE:
            final_state = env.dofbot_setInverseKine(final_pos, -1 * block_orn)
            current_pos, _ = env.get_dofbot_jointPoses()
            # 挪过去满2秒后松开
            if np.all(
                    np.isclose(np.array(current_pos),
                               np.array(final_state),
                               atol=1e-2)):
                env.dofbot_control(final_state, GRIPPER_DEFAULT_ANGLE)
            else:
                env.dofbot_control(final_state, GRIPPER_CLOSE_ANGLE)

        Reward = env.reward()

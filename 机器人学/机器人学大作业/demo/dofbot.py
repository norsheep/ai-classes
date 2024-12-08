import pybullet as p
import numpy as np
import time

# 一个功能文件


class dofbot:
    # 机械臂config
    def __init__(self, urdfPath):
        # # upper limits for null space  关节旋转角度限制
        self.ll = [-3.14, 0, -1.05, -0.16, -np.pi]
        # upper limits for null space
        self.ul = [3.14, 3.14, 4.19, 3.3, np.pi]
        # joint ranges for null space  上面的求和（就是范围）
        self.jr = [6.28, 3.14, 5.24, 3.46, 2 * np.pi]
        # rest poses for null space 机械臂五个关节的初始角度
        self.rp = [1.57, 1.57, 1.57, 1.57, 1.57]

        self.maxForce = 200.  # 机械臂关节最大扭矩
        self.fingerAForce = 2.5  # 夹爪A扭矩
        self.fingerBForce = 2.5  # 夹爪B扭矩
        self.fingerTipForce = 2  # 夹爪末端扭矩

        self.dofbotUid = p.loadURDF(
            urdfPath,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True)  # 加载机械臂模型，urdfPath模型路径，返回模型id
        # self.numJoints = p.getNumJoints(self.dofbotUid)
        self.numJoints = 5
        self.gripper_joints = [5, 6, 7, 8, 9, 10]  # 夹爪关节（index?）
        self.gripper_angle = 0  # 夹爪初始角度全部为0
        self.endEffectorPos = [0.55, 0.0, 0.6]  # 末端位置（x,y,z）

        self.jointPositions = [1.57, 1.57, 1.57, 1.57, 1.57]  # 等于初始角度

        self.motorIndices = []  # 机械臂关节索引
        for jointIndex in range(self.numJoints):
            # 五个关节
            p.resetJointState(self.dofbotUid, jointIndex,
                              self.jointPositions[jointIndex])  # 设置关节初始状态
            qIndex = p.getJointInfo(self.dofbotUid,
                                    jointIndex)[3]  # [3]返回关节位置索引qIndex
            if qIndex > -1:
                # 索引存在
                self.motorIndices.append(jointIndex)  # 添加关节索引

        for i, jointIndex in enumerate(self.gripper_joints):
            p.resetJointState(self.dofbotUid, jointIndex,
                              self.gripper_angle)  # 夹爪关节初始状态全部为0

    def reset(self):
        # 重置机械臂状态，和init一样
        self.endEffectorPos = [0.55, 0.0, 0.6]  # 重置末端位置

        self.endEffectorAngle = 0  # 重置末端角度
        self.gripper_angle = 0.0  # 重置夹爪初始角度
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.dofbotUid, jointIndex,
                              self.jointPositions[jointIndex])
        for i, jointIndex in enumerate(self.gripper_joints):
            p.resetJointState(self.dofbotUid, jointIndex, self.gripper_angle)

    def forwardKinematic(self, jointPoses):
        # 正运动学求解，输入关节角度，返回末端位置和orientation（四元数）
        for i in range(self.numJoints):
            # 从第一个关节依次根据关节角度重置该关节的位置
            p.resetJointState(self.dofbotUid,
                              jointIndex=i,
                              targetValue=jointPoses[i],
                              targetVelocity=0)
        return self.get_pose()

    def joint_control(self, jointPoses):
        # 控制机械臂关节角度（设置目标位置、速度、力矩等）
        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.dofbotUid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=200,
                                    maxVelocity=1.0,
                                    positionGain=0.3,
                                    velocityGain=1)

    def setInverseKine(self, pos, orn):
        # 逆运动学求解，输入末端位置和orientation，返回五个关节角度（有了orn可以唯一确定）
        if orn == None:
            jointPoses = p.calculateInverseKinematics(self.dofbotUid, 4, pos,
                                                      self.ll, self.ul,
                                                      self.jr, self.rp)
        else:
            jointPoses = p.calculateInverseKinematics(self.dofbotUid, 4, pos,
                                                      orn, self.ll, self.ul,
                                                      self.jr, self.rp)
        return jointPoses[:self.numJoints]  # 返回五个关节角度（无抓夹）

    def get_jointPoses(self):
        # 返回机械臂五个关节角度和夹爪角度
        jointPoses = []
        for i in range(self.numJoints + 1):
            state = p.getJointState(self.dofbotUid, i)
            jointPoses.append(state[0])
        return jointPoses[:self.numJoints], jointPoses[self.numJoints]

    def get_pose(self):
        # 返回末端位置和四元数
        state = p.getLinkState(self.dofbotUid, 4)
        pos = state[0]
        orn = state[1]
        return pos, orn

    def getObservation(self):
        # 返回观测值: 末端位置和oula角
        observation = []
        state = p.getLinkState(self.dofbotUid, 4)

        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)  # 四元数转欧拉角
        observation.extend(list(pos))
        observation.extend(list(euler))
        return observation

    def gripper_control(self, gripperAngle):
        # 夹爪控制，均使用位置控制，设置目标位置和（最大）力矩。除8为B外均为A，大小无区别：2。5
        p.setJointMotorControl2(self.dofbotUid,
                                5,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.dofbotUid,
                                6,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerBForce)
        p.setJointMotorControl2(self.dofbotUid,
                                7,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.dofbotUid,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerBForce)
        p.setJointMotorControl2(self.dofbotUid,
                                9,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.dofbotUid,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)


class Object:
    # 抓取对象config
    def __init__(self, urdfPath, block, num):
        self.id = p.loadURDF(urdfPath)
        self.half_height = 0.015 if block else 0.0745
        self.num = num

        self.block = block

    def reset(self):

        if self.num == 1:
            p.resetBasePositionAndOrientation(
                self.id, np.array([0.20, 0.1, self.half_height]),
                p.getQuaternionFromEuler([0, 0, np.pi / 6]))
        else:
            p.resetBasePositionAndOrientation(
                self.id, np.array([0.2, -0.1, 0.005]),
                p.getQuaternionFromEuler([0, 0, 0]))

    def pos_and_orn(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        # euler = p.getEulerFromQuaternion(quat)
        return pos, orn


def check_pairwise_collisions(bodies):
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and \
                    len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0:
                return True
    return False


class DofbotEnv:
    # 实验仿真环境
    def __init__(self):
        self._timeStep = 0.001
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1.0, 90, -40, [0, 0, 0])
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)  # 设置重力

        p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF("models/table_collision/table.urdf", [0.5, 0, -0.625],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._dofbot = dofbot(
            "models/dofbot_urdf_with_gripper/dofbot_with_gripper.urdf")
        self._object1 = Object("models/box_green.urdf", block=True, num=1)
        self._object2 = Object("models/box_purple.urdf", block=True, num=2)
        self.target_pos = np.array([0.2, -0.1, 0.015])  # 感觉不对，自行修改了一下

    def reset(self):

        self._object1.reset()
        self._object2.reset()
        self._dofbot.reset()
        p.stepSimulation()

    def dofbot_control(self, jointPoses, gripperAngle):
        '''
        :param jointPoses: 数组，机械臂五个关节角度
        :param gripperAngle: 浮点数，机械臂夹爪角度，负值加紧，真值张开
        :return:
        '''

        self._dofbot.joint_control(jointPoses)
        self._dofbot.gripper_control(gripperAngle)
        p.stepSimulation()
        time.sleep(self._timeStep)

    def dofbot_setInverseKine(self, pos, orn=None):
        '''

        :param pos: 机械臂末端位置，xyz
        :param orn: 机械臂末端方向，四元数
        :return: 机械臂各关节角度
        '''
        jointPoses = self._dofbot.setInverseKine(pos, orn)
        return jointPoses

    def dofbot_forwardKine(self, jointStates):
        return self._dofbot.forwardKinematic(jointStates)

    def get_dofbot_jointPoses(self):
        '''
        :return: 机械臂五个关节位置+夹爪角度
        '''
        jointPoses, gripper_angle = self._dofbot.get_jointPoses()

        return jointPoses, gripper_angle

    def get_dofbot_pose(self):
        '''
        :return: 机械臂末端位姿，xyz+四元数
        '''
        pos, orn = self._dofbot.get_pose()
        return pos, orn

    def get_block_pose(self):
        '''
        :return: 物块位姿，xyz+四元数
        '''
        pos, orn = self._object1.pos_and_orn()
        return pos, orn

    def get_target_pose(self):
        '''
        :return: 目标位置，xyz
        '''
        return self.target_pos

    def reward(self):
        '''
        :return: 是否完成抓取放置
        '''
        pos, orn = self._object1.pos_and_orn()
        dist = np.sqrt((pos[0] - self.target_pos[0])**2 +
                       (pos[1] - self.target_pos[1])**2)
        if dist < 0.01 and pos[2] < 0.02:
            return True
        return False

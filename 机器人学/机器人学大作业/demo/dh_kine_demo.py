import roboticstoolbox as rtb
import numpy as np
import math

pi = 3.1415926  # 定义pi常数

l1 = 0.104  # 定义第一连杆长度
l2 = 0.08285  # 定义第三连杆长度
l3 = 0.08285  # 定义第四连杆长度
l4 = 0.12842  # 定义第五连杆长度

# student version
# 用改进DH参数发表示机器人正运动学
# TODO: modify the dh param
dofbot = rtb.DHRobot([
    rtb.RevoluteMDH(a=0, alpha=0, d=l1, offset=0),
    rtb.RevoluteMDH(a=0, alpha=pi / 2, d=0, offset=pi / 2),
    rtb.RevoluteMDH(a=l2, alpha=0, d=0, offset=0),
    rtb.RevoluteMDH(a=l3, alpha=0, d=0, offset=pi / 2),
    rtb.RevoluteMDH(a=0, alpha=pi / 2, d=l4, offset=0)
])
# object，该类有正逆运动学求解函数？

# 输出机器人DH参数矩阵
print(dofbot)
'''
Part1 给出一下关节姿态时的机械臂正运动学解，并附上仿真结果
0.(demo) [0., pi/3, pi/4, pi/5, 0.]
1.[pi/2, pi/5, pi/5, pi/5, pi]
2.[pi/3, pi/4, -pi/3, -pi/4, pi/2]
3.[-pi/2, pi/3, -pi/3*2, pi/3, pi/3]
'''
# part1 demo
fkine_input0 = [0., pi / 3, pi / 4, pi / 5, 0.]
fkine_result0 = dofbot.fkine(fkine_input0)
print('demo0:')
print(fkine_result0)
dofbot.plot(q=fkine_input0, block=True)

# part1-1
fkine_input1 = [pi / 2, pi / 5, pi / 5, pi / 5, pi]
fkine_result1 = dofbot.fkine(fkine_input1)
print('part1:')
print(fkine_result1)
dofbot.plot(q=fkine_input1, block=True)
# part1-2
fkine_input2 = [pi / 3, pi / 4, -pi / 3, -pi / 4, pi / 2]
fkine_result2 = dofbot.fkine(fkine_input2)
print('part2:')
print(fkine_result2)
dofbot.plot(q=fkine_input2, block=True)
# part1-3
fkine_input3 = [-pi / 2, pi / 3, -pi / 3 * 2, pi / 3, pi / 3]
fkine_result3 = dofbot.fkine(fkine_input3)
print('part3:')
print(fkine_result3)
dofbot.plot(q=fkine_input3, block=True)
'''
Part2 给出一下关节姿态时的机械臂逆运动学解，并附上仿真结果
0.(demo) 
    [
        [-1., 0., 0., 0.1,],
        [0., 1., 0., 0.],
        [0., 0., -1., -0.1],
        [0., 0., 0., 1.]
    ]
1.
    [
        [1., 0., 0., 0.1,],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.1],
        [0., 0., 0., 1.]
    ]
2.
    [
        [cos(pi/3), 0., -sin(pi/3), 0.2,],
        [0., 1., 0., 0.],
        [sin(pi/3), 0., cos(pi/3)., 0.2],
        [0., 0., 0., 1.]
    ]
3.
    [
        [-0.866, -0.25, -0.433, -0.03704,],
        [0.5, -0.433, -0.75, -0.06415],
        [0., -0.866, 0.5, 0.3073],
        [0., 0., 0., 1.]
    ]
'''
# part2 逆运动学求解，输出为theta=5个角度
#part2 demo
target_pos0 = np.array([[
    -1.,
    0.,
    0.,
    0.1,
], [0., 1., 0., 0.], [0., 0., -1., -0.1], [0., 0., 0., 1.]])
ikine_result0 = dofbot.ik_LM(target_pos0)[0]
print("ikine0: ", np.array(ikine_result0) / pi * 180)
dofbot.plot(q=ikine_result0, block=True)
# 正运动学检测答案正确性
# fkine_input20 =
# fkine_result20 = dofbot.fkine(fkine_input20)
# print("check demo2-0\n", fkine_result20)

#part2-1
target_pos1 = np.array([[
    1.,
    0.,
    0.,
    0.1,
], [0., 1., 0., 0.], [0., 0., 1., 0.1], [0., 0., 0., 1.]])
ikine_result1 = dofbot.ik_LM(target_pos1)[0]
print("ikine1: ", np.array(ikine_result1) / pi * 180)
dofbot.plot(q=ikine_result1, block=True)
# 正运动学检测答案正确性
# fkine_input21 =
# fkine_result21 = dofbot.fkine(fkine_input21)
# print("check 2-1\n", fkine_result21)

#part2-2
target_pos2 = np.array([[
    np.cos(pi / 3),
    0.,
    np.sin(-pi / 3),
    0.2,
], [0., 1., 0., 0.], [np.sin(pi / 3), 0.,
                      np.cos(pi / 3), 0.2], [0., 0., 0., 1.]])
ikine_result2 = dofbot.ik_LM(target_pos2, q0=[0, 0, 0, 0, 0])[0]
print("ikine2: ", np.array(ikine_result2) / pi * 180)
dofbot.plot(q=ikine_result2, block=True)
# 正运动学检测答案正确性
# fkine_input22 =
# fkine_result22 = dofbot.fkine(fkine_input22)
# print("check 2-2\n", fkine_result22)

#part2-3
target_pos3 = np.array([[
    -0.866,
    -0.25,
    -0.433,
    -0.03704,
], [0.5, -0.433, -0.75, -0.06415], [0., -0.866, 0.5, 0.3073], [0., 0., 0.,
                                                               1.]])
ikine_result3 = dofbot.ik_LM(target_pos3, q0=[0, 0, 0, 0, 0])[0]
print("ikine3: ", np.array(ikine_result3) / pi * 180)
dofbot.plot(q=ikine_result3, block=True)
# 正运动学检测答案正确性
# fkine_input23 =
# fkine_result23 = dofbot.fkine(fkine_input23)
# print("check 2-3\n", fkine_result23)

#part 3 ：绘制机械臂工作空间
import matplotlib.pyplot as plt

joint_limits = [
    [-pi, pi],
    [-pi / 2, pi / 2],
    [-5 * pi / 6, 5 * pi / 6],
    [-5 * pi / 9, 5 * pi / 9],
    [-pi, pi],
]

num_sample = 1000
pos_array = np.zeros((num_sample, 3))

for index in range(num_sample):
    #均匀采样角度后选取末端位置作为工作空间上的点
    coordinates = [
        np.random.uniform(joint_limits[i][0], joint_limits[i][1])
        for i in range(5)
    ]
    fkine_result = dofbot.fkine(coordinates)  # 正运动学求解
    pos_array[index] = np.array(fkine_result)[0:3, 3]  # 保存末端位置

#开画
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]
sc = ax.scatter(x, y, z, s=20, c=z, cmap='viridis', marker='o')
plt.colorbar(sc)
plt.show()

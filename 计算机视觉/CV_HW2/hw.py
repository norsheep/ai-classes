import cv2
import numpy as np
import glob

# cv2.imread(), cv2.imwrite(), cv2.findChessboardCorners()可以用
'''
1.Define real world coordinates of 3D points using checkerboard pattern of known size. 按照棋盘已知大小，定义真实世界坐标
2.Capture the images of the checkerboard from different viewpoints. 不同视角的棋盘图片
3.Find chessboard corners as well as finding the pixel coordinates (u, v) for each 3D point in
different images  找到棋盘角点,以及在不同图像中找到每个3D点的像素坐标
4.Find camera parameters, the 3D points, and the pixel coordinates using linear algebra. 
使用线性代数,找到相机参数,3D点和像素坐标

最终结果包含
1. The camera's intrinsic parameters, including the focal length, principal point (distortion
parameters are not required.) 相机内参,包括焦距,主点
2. The projection error corresponding to each image (which is the error after projecting the 3D
points back onto the pixel plane; the smaller the error, the more accurate the estimation of the
intrinsic parameters). 每个图像对应的投影误差(将3D点投影回像素平面后的误差;误差越小,对内参的估计越准确)

注意事项
1.For simplification, you can pass the points in XY plane as (0,0), (1,0), (2,0), ... which denotes 
the location of points. In this case, the results will be in the scale of size of chess board 
square. 结果在棋盘方格的大小范围内时,可以用XY:(0,0),(1,0),(2,0)简化表示点的位置
2.To minimize the impact of the distortion parameters, please try to calculate the intrinsic 
parameters using the central region of the images as much as possible. 
尽可能使用图像的中心区域计算内参,以最小化畸变参数的影响
'''


def read_images(image_directory):
    # Read all jpg images from the specified directory  shape (n, h, w, c)
    return [
        cv2.imread(image_path)
        for image_path in glob.glob(f"{image_directory}/*.jpg")
    ]


def find_image_points(images, pattern_size, center=False):
    world_points = []
    image_points = []

    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        width, height = pattern_size  # 31,23,the inner corner of the chessboard
        return np.mgrid[0:width, 0:height].T.reshape(-1, 2).astype(np.float32)

    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        flag, corners = cv2.findChessboardCorners(image, pattern_size,
                                                  None)  # 返回坐标N(w,h)格式
        if flag:
            # debug —— only 4 valid images
            # cv2.drawChessboardCorners(image, pattern_size, corners, flag)
            # cv2.imshow('Detected Corners', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # 和世界坐标系对齐
            return corners.reshape(-1, 2).astype(np.float32)
        else:
            return None

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    image_size = images[0].shape[:2][::-1]  # 图像大小 (w,h)
    center_point = np.array(image_size) / 2
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            image_point = corners
            world_point = init_world_points(pattern_size)
            if center:
                # just get center n points
                index = np.argsort(
                    np.linalg.norm(image_point - center_point, axis=1))[:300]
                world_point, image_point = world_point[index], image_point[
                    index]
            image_points.append(image_point)
            world_points.append(world_point)

    return world_points, image_points


def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(
        image_points
    ), "The number of world coordinates and image coordinates must match"

    num = len(world_points)  # The number of images
    A = []
    B = []
    K = np.zeros(
        (3, 3)
    )  # Camera calibration matrix # i do some change: instrinct matrix is 3*4(0,0,0)
    P = None

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    n = len(world_points[0])  # 一张图片的点数
    G = np.zeros((2 * n, 6))  # 2n*9
    P = []  # 保存每张图片的投影矩阵H
    for i in range(num):
        world_point = world_points[i]  # 一张图片的世界坐标集合
        image_point = image_points[i]  # 一张图片的图像坐标集合

        # Construct the matrix A
        A = np.zeros((2 * n, 9))  # 2n*9
        for j in range(n):
            x, y = world_point[j]
            u, v = image_point[j]
            A[2 * j] = np.array([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A[2 * j + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]

        # Solve matrix H from A
        eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
        h = eigenvectors[:, np.argmin(eigenvalues)]
        h11, h12, h13, h21, h22, h23, h31, h32, h33 = h
        H = h.reshape(3, 3)
        P.append(H)

        # construct matrix R from H to calculate B, B=K^(-T) * K^(-1)
        G[2 * i] = np.array([
            h11 * h12, h11 * h22 + h21 * h12, h11 * h32 + h31 * h12, h21 * h22,
            h21 * h32 + h31 * h22, h31 * h32
        ])
        G[2 * i + 1] = np.array([
            h11 * h11 - h12 * h12, 2 * (h11 * h21 - h12 * h22),
            2 * (h11 * h31 - h32 * h12), h21 * h21 - h22 * h22,
            2 * (h31 * h21 - h32 * h22), h31 * h31 - h32 * h32
        ])

    # Solve B for K and R
    eigenvalues, eigenvectors = np.linalg.eig(G.T @ G)
    b = eigenvectors[:, np.argmin(eigenvalues)]
    B = np.array([b[0], b[1], b[2], b[1], b[3], b[4], b[2], b[4],
                  b[5]]).reshape(3, 3)
    # Please ensure that the diagonal elements of K are positive

    # Decompose B to get K
    K = np.linalg.cholesky(B).T  # 分解得到的是K^(-1)，转置回上三角
    # R = KH for every H
    K = np.linalg.inv(K)  # K
    K = K / K[2, 2]  # 已知K[2,2]为1，进行归一化

    return K, P


# Main process
image_path = 'Sample_Calibration_Images'
images = read_images(image_path)  # 返回一个图片列表
print(images.__len__())  # 25

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
# 查方格个数  32 24   31*23 = 713
pattern_size = (31, 23)  # The inner corner of the chessboard

world_points, image_points = find_image_points(images, pattern_size,
                                               True)  # 返回世界坐标和图像坐标
camera_matrix, camera_extrinsics = calibrate_camera(
    world_points, image_points)  # 返回相机内参和所有图片的投影矩阵（计算外参为K^(-1)H）

# 打印debug项
#print(images[0].shape)  # (480, 640, 3)

np.set_printoptions(suppress=True)
print("The number of used corners closed to center:",
      world_points[0].shape[0])  # (713, 2)
# 打印相机内参
print("Camera Calibration Matrix:")
print(camera_matrix)


def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded. 可选，不算分
    # return None, directly print the results
    # TODO:
    images = read_images(image_directory)
    world_points, image_points = find_image_points(images, pattern_size, True)
    expand_vector = np.zeros((world_points[0].shape[0], 1), dtype=np.float32)
    object_points = [
        np.append(view, expand_vector, axis=1) for view in world_points
    ]
    image_size = images[0].shape[:2][::-1]
    _, camera_matrix, _, _, _ = cv2.calibrateCamera(object_points,
                                                    image_points, image_size,
                                                    None, None)
    print("Camera Matrix:\n", camera_matrix)


def reprojection_error(world_points, image_points, camera_matrix):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image 重投影误差
    error_list = []
    image_num = len(world_points)
    for i in range(image_num):
        world_point = world_points[i]
        image_point = image_points[i]
        world_point = np.hstack(
            (world_point, np.ones((world_point.shape[0], 1),
                                  dtype=np.float32)))
        projection_point = camera_matrix[i] @ world_point.T
        projection_point = projection_point / projection_point[2]
        projection_point = projection_point[:2].astype(np.float32)
        error = cv2.norm(image_point, projection_point.T, cv2.NORM_L2) / len(
            image_point)  # 计算误差
        error_list.append(error)
    return error_list


print("Camera Calibration Matrix by OpenCV:")
test(image_path, pattern_size)
world_points, image_points = find_image_points(
    images, pattern_size
)  # all corner points. If only check points for caluculate K&H, set center=True
images_error = reprojection_error(world_points, image_points,
                                  camera_extrinsics)
print("Reprojection Error:", images_error)

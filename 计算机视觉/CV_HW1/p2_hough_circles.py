#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.signal import convolve2d


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    # define sobel operator of x and y direction
    sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_operator_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # calculate gradient in x and y direction juanji
    gradient_x = convolve2d(image,
                            sobel_operator_x,
                            mode='same',
                            boundary='fill',
                            fillvalue=0)
    gradient_y = convolve2d(image,
                            sobel_operator_y,
                            mode='same',
                            boundary='fill',
                            fillvalue=0)
    # calculate magnitude of gradient
    edge_image = np.sqrt(gradient_x**2 + gradient_y**2)
    # normalize edge_image
    edge_image = edge_image / np.max(edge_image) * 255
    return edge_image


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """

    # threshold edge image, return a bool array(no need to print)
    thresh_edge_image = (edge_image >= edge_thresh)
    # initialize accumulator array
    accum_array = np.zeros(
        (len(radius_values), edge_image.shape[0], edge_image.shape[1]))
    # find circles
    for i in range(edge_image.shape[1]):
        for j in range(edge_image.shape[0]):
            if thresh_edge_image[j][i]:
                for r in range(len(radius_values)):
                    # By hough transform, calculate the center of the circle, totally 360 points(posible center)
                    for degree in range(360):
                        theta = np.deg2rad(degree)
                        a = int(i - radius_values[r] * np.cos(theta))
                        b = int(j - radius_values[r] * np.sin(theta))
                        if a >= 0 and a < edge_image.shape[
                                1] and b >= 0 and b < edge_image.shape[0]:
                            # add vote
                            accum_array[r][b][a] += 1
    return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """

    # find circles
    all_circles = []  # store all circles
    circle_image = np.copy(image)  # copy image for drawing circles
    # find circles by hough threshold
    for r in range(accum_array.shape[0]):
        for j in range(accum_array.shape[1]):
            for i in range(accum_array.shape[2]):
                if accum_array[r][j][i] >= hough_thresh:
                    all_circles.append((radius_values[r], j, i))

    # remove_duplicated circles: if center of two circles are too close, remove it
    circles = []  # store circles without duplicated
    min_dist = 10  # set a threshold to remove duplicated circles
    for circle in all_circles:
        r, y, x = circle
        if not any(
                np.sqrt((x - ux)**2 + (y - uy)**2) < min_dist
                for ur, uy, ux in circles):
            circles.append(circle)
            cv2.circle(circle_image, (x, y), r, (0, 255, 0), 2)

    return circles, circle_image


if __name__ == '__main__':

    image_name = 'coins'
    edge_thresh = 120  # 0~255 threshold:120
    hough_thresh = 183  # 0~266 threshold:183
    radius_values = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20]

    image = cv2.imread('data/' + image_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge_img = detect_edges(gray_image)
    # find edge threshold
    print(max(edge_img.flatten()), min(edge_img.flatten()))  # 0.0 255.0 (781)
    thresh_edge_img, accum_array = hough_circles(edge_img, edge_thresh,
                                                 radius_values)
    # find hough threshold
    print(max(accum_array.flatten()), min(accum_array.flatten()))  #331 0
    circles, circle_img = find_circles(image, accum_array, radius_values,
                                       hough_thresh)
    cv2.imwrite('output/' + image_name + '_edge.png',
                edge_img)  # function detect_edges return edge_img
    cv2.imwrite('output/' + image_name + '_circle.png',
                circle_img)  #function find_circles return circle_img

    print(accum_array[0][:10]
          [:10])  # all zero, and another output is a bool array
    print(circles)
    # [(29, 32, 147), (29, 69, 215), (29, 105, 35), (29, 118, 173), (29, 145, 94), (29, 207, 120), (25, 49, 55), (25, 100, 264), (25, 171, 235), (24, 83, 109)]
    print(len(circles))  # 10
    print('done')

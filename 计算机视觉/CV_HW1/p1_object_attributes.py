#!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
    # 255 if intensity >= thresh_val else 0
    binary_image = np.zeros_like(gray_image)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            binary_image[i][j] = 255 if gray_image[i][j] >= thresh_val else 0
    return binary_image


def label(binary_image):
    # label connected components
    labeled_image = np.zeros_like(binary_image, dtype=np.int32)
    label_dict = {
    }  # a dictionary to store equivalent relationship, key: label, value: root label
    label_dict_num = 0  # a counter to give label to each connected component
    a, b = binary_image.shape  # a: height, b: width

    # first pass to give label to each pixel and address equivalent relationship in a dictionary
    for j in range(a):
        for i in range(b):
            # find 4-neighbours（up, left, up-left, up-right）
            neighbours = []
            if binary_image[j][i] == 255:
                if j - 1 >= 0 and binary_image[j - 1][i] == 255:
                    neighbours.append(labeled_image[j - 1][i])
                if i - 1 >= 0 and binary_image[j][i - 1] == 255:
                    neighbours.append(labeled_image[j][i - 1])
                if i - 1 >= 0 and j - 1 >= 0 and binary_image[j - 1][i -
                                                                     1] == 255:
                    neighbours.append(labeled_image[j - 1][i - 1])
                if i + 1 < b and j - 1 >= 0 and binary_image[j - 1][i +
                                                                    1] == 255:
                    neighbours.append(labeled_image[j - 1][i + 1])

                if neighbours:
                    # if neighbours, assign the minimal label to the pixel
                    min_label = min(neighbours)
                    labeled_image[j][i] = min_label
                    for neighbour in neighbours:
                        if neighbour != min_label:
                            neighbour_root = get_root(label_dict, neighbour)
                            min_root = get_root(label_dict, min_label)
                            if neighbour_root != min_root:
                                label_dict[neighbour_root] = min_root
                else:
                    # if no neighbour, assign a new label
                    label_dict_num += 1
                    labeled_image[j][i] = label_dict_num
                    label_dict[label_dict_num] = label_dict_num

    # second pass to update the label
    for i in range(b):
        for j in range(a):
            if labeled_image[j][i] != 0:
                # use a linear transformation to make the image more clear and different
                labeled_image[j][i] = 128 + (
                    10 * get_root(label_dict, labeled_image[j][i])) % 128

    return labeled_image


def get_root(label_dict, label):
    # a compliment function to get the root of a label(find minimal label)
    while label_dict[label] != label:
        label = label_dict[label]
    return label


def get_attribute(labeled_image):
    # compute a list of object attributes(position, otrientation, roundedness)
    attribute_list = []  # a list to store attributes
    a, b = labeled_image.shape  # a: height, b: width
    param = {
    }  # a dictionary to store parameters of each connected component, key: label, value: parameters
    for i in range(b):
        for j in range(a):
            if labeled_image[j][i] != 0:
                # initialize parameters
                if labeled_image[j][i] not in param:
                    param[labeled_image[j][i]] = {
                        'x': 0,
                        'y': 0,
                        'a': 0,
                        'b': 0,
                        'c': 0,
                        'num': 0
                    }
                # calculate parameters
                param[labeled_image[j][i]]['x'] += i
                param[labeled_image[j][i]]['y'] += a - j - 1
                param[labeled_image[j][i]]['a'] += i**2
                param[labeled_image[j][i]]['c'] += (a - j - 1)**2
                param[labeled_image[j][i]]['b'] += 2 * i * (a - j - 1)
                param[labeled_image[j][i]]['num'] += 1

    # calculate attributes
    for key, value in param.items():
        dict = {}
        # calculate position
        x_bar = value['x'] / value['num']
        y_bar = value['y'] / value['num']
        dict['position'] = {'x': x_bar, 'y': y_bar}
        # calculate arctan
        value['a'] = value[
            'a'] - 2 * x_bar * value['x'] + value['num'] * x_bar**2
        value['b'] = value['b'] - 2 * x_bar * value['y'] - 2 * y_bar * value[
            'x'] + 2 * value['num'] * x_bar * y_bar
        value['c'] = value[
            'c'] - 2 * y_bar * value['y'] + value['num'] * y_bar**2
        theta1 = 0.5 * np.arctan(value['b'] /
                                 (value['a'] - value['c'])) % np.pi
        theta2 = (theta1 + np.pi / 2) % np.pi
        # calculate orientation and roundedness
        E1 = value['a'] * np.sin(theta1)**2 - value['b'] * np.sin(
            theta1) * np.cos(theta1) + value['c'] * np.cos(theta1)**2
        E2 = value['a'] * np.sin(theta2)**2 - value['b'] * np.sin(
            theta2) * np.cos(theta2) + value['c'] * np.cos(theta2)**2
        # print('b',value['b'])
        # print(theta1, theta2)
        # print(E1, E2)
        dict['orientation'] = theta2 if E1 > E2 else theta1
        dict['roundedness'] = E2 / E1 if E1 > E2 else E1 / E2
        attribute_list.append(dict)
    return attribute_list


def main(argv):
    img_name = argv[0]  # picture name
    thresh_val = int(argv[1])  # threshold value
    img = cv2.imread('data/' + img_name + '.png',
                     cv2.IMREAD_COLOR)  # H x W x 3
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # H x W
    # cv2.imshow('gray', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(gray_image.shape)

    binary_image = binarize(gray_image, thresh_val=thresh_val)  # H x W
    # print(binary_image.shape)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)

    # save images
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_binary.png",
                binary_image)  # function binarize return value
    cv2.imwrite('output/' + img_name + "_labeled.png",
                labeled_image)  # function label return value
    print(attribute_list)  # function get_attribute return value


if __name__ == '__main__':
    main(sys.argv[1:])
    # 1: jump the filename

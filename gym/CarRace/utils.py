import cv2
from cv2 import resize, COLOR_BGR2GRAY, cvtColor
import numpy as np
import itertools
import tensorflow as tf


def pre_processing(img):
    processed = img[0:80, 0:98]
    processed = resize(processed, (28, 28))
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    imask_green = mask_green > 0
    green = np.zeros_like(processed, np.uint8)
    green[imask_green] = processed[imask_green]
    processed = green
    processed[np.where((processed == [102, 229, 102]).all(axis=2))] = [102, 204, 102]
    processed[np.where((processed == [0, 255, 0]).all(axis=2))] = [0, 0, 0]
    processed = cvtColor(processed, COLOR_BGR2GRAY)
    processed = processed / 255
    processed = np.array(processed)
    return processed


def scale_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    return img


def get_sensors_pic(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    speed = img[85:94, 12:14]
    gyro = img[87:91, 58:86]
    abs_info1 = img[85:94, 17:19]
    abs_info2 = img[85:94, 19:21]
    abs_info3 = img[85:94, 22:23]
    abs_info4 = img[85:94, 24:26]
    return speed, gyro, abs_info1, abs_info2, abs_info3, abs_info4


def get_sensor_data(img, column_or_row):
    if column_or_row == "row":
        row_united = list(itertools.chain.from_iterable(img[:, img.shape[1] // 2:img.shape[1] // 2 + 1]))
        data = len(row_united) - row_united.count(0)
        data = data / len(row_united)
        return data
    elif column_or_row == "column":
        column_united = list(itertools.chain.from_iterable(img[img.shape[0] // 2:img.shape[0] // 2 + 1:, :]))
        sign = 0
        if column_united[13] != 0:
            sign = -1
        elif column_united[14] != 0:
            sign = 1
        data = ((len(column_united) - column_united.count(0)) * 2 * sign) / len(column_united)
        return data


def get_sensors_data_from_images(speed, gyro, abs1, abs2, abs3, abs4):
    speed_data = get_sensor_data(speed, "row")
    gyro_data = get_sensor_data(gyro, "column")
    abs1_data = get_sensor_data(abs1, "row")
    abs2_data = get_sensor_data(abs2, "row")
    abs3_data = get_sensor_data(abs3, "row")
    abs4_data = get_sensor_data(abs4, "row")
    return speed_data, gyro_data, abs1_data, abs2_data, abs3_data, abs4_data


def add_sensors_data_to_observation(observation, img):
    sensors_pics = get_sensors_pic(img)
    sensors = tf.convert_to_tensor([list(get_sensors_data_from_images(sensors_pics[0], sensors_pics[1], sensors_pics[2],
                                                                        sensors_pics[3], sensors_pics[4], sensors_pics[5]))], dtype=tf.float32)
    observation = tf.concat([observation, sensors], 1)
    return observation

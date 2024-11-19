import cv2
import numpy as np
from PIL import Image
from .micaps_data_io import read_micaps_11


def compute_rgb(speed, direction, only_r=False):
    """

    :param only_r: only R channel
    :param speed: wind speed
    :param direction: wind direction
    :return: image：image after three channels combined
    """
    row, col = speed.shape

    result_image = np.zeros((row, col, 3), dtype=np.uint8)

    result_image[..., 0] = np.where(speed < 14, 0, 255 / (1 + np.exp(-speed + 20)))

    if only_r:
        result_image[..., 1] = np.zeros((row, col), dtype=np.uint8)
    else:
        result_image[..., 1] = ((np.cos(2 * np.pi * direction / 359) + 1) * 127.5).astype(np.uint8)

    if only_r:
        result_image[..., 2] = np.zeros((row, col), dtype=np.uint8)
    else:
        result_image[..., 2] = ((np.sin(2 * np.pi * direction / 359) + 1) * 127.5).astype(np.uint8)

    MaskImage = np.where(speed < 14, 0, 1)
    result_image[..., 0] = result_image[..., 0] * MaskImage
    result_image[..., 1] = result_image[..., 1] * MaskImage
    result_image[..., 2] = result_image[..., 2] * MaskImage

    image = Image.fromarray(result_image)
    return image


def compute_speed_and_direction(array):
    row, col, _ = array.shape

    u_array = array[:, :, 0]
    v_array = array[:, :, 1]

    speed = np.sqrt(u_array ** 2 + v_array ** 2)

    speed = np.where(speed >= 40, 40, speed)

    direction = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            u_value, v_value = u_array[i, j], v_array[i, j]

            if u_value == 0 and v_value >= 0:
                direction[i, j] = 0
            elif u_value > 0 and v_value == 0:
                direction[i, j] = 90
            elif u_value < 0 and v_value == 0:
                direction[i, j] = 270
            elif u_value == 0 and v_value < 0:
                direction[i, j] = 180
            elif u_value > 0 and v_value > 0:
                direction[i, j] = np.arctan(u_value / v_value) * 180 / np.pi
            elif u_value < 0 < v_value:
                direction[i, j] = np.arctan(-v_value / u_value) * 180 / np.pi + 270
            elif u_value < 0 and v_value < 0:
                direction[i, j] = np.arctan(u_value / v_value) * 180 / np.pi + 180
            elif u_value > 0 > v_value:
                direction[i, j] = np.arctan(-v_value / u_value) * 180 / np.pi + 90

    return speed, direction


def generate_image(filepath, image_size):
    stacked_array, lon_range, lat_range = read_micaps_11(filepath)
    resized_array = cv2.resize(stacked_array, image_size)
    speed, direction = compute_speed_and_direction(resized_array)
    image = compute_rgb(speed, direction)
    return image


def generate_image_only_r(filepath, image_size):
    stacked_array, lon_range, lat_range = read_micaps_11(filepath)
    resized_array = cv2.resize(stacked_array, image_size)
    speed, direction = compute_speed_and_direction(resized_array)
    image = compute_rgb(speed, direction, only_r=True)
    return image

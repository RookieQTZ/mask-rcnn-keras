# encoding: utf-8
"""
@author: _Jack Sparrow
@time:  2022/7/28 16:35
@file: mask_to_json1.py
@desc: 掩码图片转json
"""

import cv2
import pandas as pd
import os


def get_coor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变为灰度图
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  ## 阈值分割得到二值化图片
    # cv2.namedWindow('binary', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('binary', binary)
    # cv2.waitKey(0)
    contours, heriachy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pointsList = []
    print(len(contours))  # 目标个数
    for i, contour in enumerate(contours):
        # print(heriachy)
        if len(contour) < 20:
            continue
        num = len(contour[:, 0, 0])  # 个数
        hundred = num // 80  # 每个点之间的步长
        tem = contour[:, 0][::hundred]
        return tem, 1


if __name__ == '__main__':
    img_path = r"../data/cv2_mask/mask1.png"
    img = cv2.imread(img_path)
    h, w = img.shape[:-1]
    contours, flag = get_coor(img)
    i = 1
    # 坐标数组contours
    color = (0, 0, 255)
    # circle(图片，中心点(x, y), 半径长度, 颜色，圆边厚度)
    for x, y in contours:
        cv2.circle(img, (x, y), 1, color, 3)
        cv2.imwrite("result.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
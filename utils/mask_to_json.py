import datetime
import json
import os
import io
import re
import fnmatch
import json
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from PIL import Image
import base64
from base64 import b64encode
import cv2

ROOT_DIR = os.path.join('..', 'data')
IMAGE_DIR = os.path.join(ROOT_DIR, "pic")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "cv2_mask")
JSON_DIR = os.path.join(ROOT_DIR, "json")


def img_tobyte(img_pil):
    # 类型转换 重要代码
    # img_pil = Image.fromarray(roi)
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string


# json
def draw_mask_point(json_file, img_path):
    # 从json中拿到轮廓信息
    with open(json_file, 'r') as load_f:
        mask_json = json.load(load_f)
    contours = []
    shapes = mask_json['shapes']
    for i, _ in enumerate(shapes):
        points = shapes[i]['points']
        for x, y in points:
            contours.append((x, y))

    img = cv2.imread(img_path)
    color = (0, 0, 255)
    # circle(图片，中心点(x, y), 半径长度, 颜色，圆边厚度)
    # x, y 浮点型 -> 整形
    for x, y in contours:
        cv2.circle(img, (int(x), int(y)), 1, color, 3)
        cv2.imwrite("result.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 批处理图片=========================实际没有用，之后优化
CACHE_SIZE = 50
img_cache = {}
json_cache = {}

annotation_files = os.listdir(ANNOTATION_DIR)
for annotation_filename in annotation_files:
    coco_output = {
        "version": "3.16.7",
        "flags": {},
        "fillColor": [255, 0, 0, 128],
        "lineColor": [0, 255, 0, 128],
        "imagePath": {},
        "shapes": [],
        "imageData": {}}

    print(annotation_filename)
    class_id = 1
    name = annotation_filename.split('.', 3)[0]
    name1 = name + '.jpg'
    coco_output["imagePath"] = name1

    # 缓存原图及标注图
    # img_cache[name] = (Image.open(IMAGE_DIR + '/' + name1), Image.open(ANNOTATION_DIR + '/' + annotation_filename))

    image = Image.open(os.path.join(IMAGE_DIR, name1))
    # 每50一批数据、最后一批数据
    imageData = img_tobyte(image)
    coco_output["imageData"] = imageData

    binary_mask = np.asarray(Image.open(os.path.join(ANNOTATION_DIR, annotation_filename))
                             .convert('1')).astype(np.uint8)
    segmentation = pycococreatortools.binary_mask_to_polygon(binary_mask, tolerance=2)  # 采样点数目
    # 筛选多余的点集合
    for item in segmentation:
        if (len(item) > 10):

            list1 = []

            for i in range(0, len(item), 2):
                list1.append([item[i], item[i + 1]])

            seg_info = {'points': list1, "fill_color": 'null', "line_color": 'null', "label": "1",
                        "shape_type": "polygon", "flags": {}}
            coco_output["shapes"].append(seg_info)
    coco_output["imageHeight"] = binary_mask.shape[0]
    coco_output["imageWidth"] = binary_mask.shape[1]

    full_path = os.path.join('{}', name + '.json')

    with open(full_path.format(JSON_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    # 画出轮廓
    # draw_mask_point(full_path.format(JSON_DIR), os.path.join(ANNOTATION_DIR, annotation_filename))

import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('./save/DFDCmodel')
print(model)
directory_path_validate = "C:/Users/12136/Downloads/9.jpg"

def load_and_process_image(image_path, target_size=(128, 128)):
    """
    加载图像，如果图像是RGBA则转换为RGB，然后调整到目标尺寸并归一化。
    """
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize(target_size)
    image_array = np.array(resized_image) / 255
    return image_array


def check_and_adjust_image(image, target_size=(128, 128, 3)):
    """
    检查图像尺寸并进行调整。
    """
    if image.shape == target_size:
        return image
    else:
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize(target_size)
        return np.array(resized_image)


def calculate_accuracy(model, image_path):
    """
    接受包含图像路径的列表，对图像进行预测并返回准确率。
    """
    validate_array = []
    image_array = load_and_process_image(image_path)
    adjusted_image = check_and_adjust_image(image_array)
    validate_array.append(adjusted_image)

    validate_array = np.array(validate_array)
    predictions = model.predict(validate_array)

    return 1-predictions[0], predictions[0]

ans,ans1=calculate_accuracy(model, directory_path_validate)
print(ans)
print(ans1)


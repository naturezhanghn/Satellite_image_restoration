import cv2
import numpy as np

# 读取图像
img = cv2.imread("imga.tif")

# 将图像数据类型转换为 float32，便于处理
img = np.float32(img)

# 线性对比度拉伸
min_val, max_val = np.percentile(img, [2, 100])
img_stretched = 255 * (img - min_val) / (max_val - min_val)
img_stretched = np.clip(img_stretched, 0, 255)

# 将数据类型转换回 uint8
img_stretched = np.uint8(img_stretched)

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
if img_stretched.shape[2] == 3:
    for i in range(3):
        img_stretched[:, :, i] = clahe.apply(img_stretched[:, :, i])
else:
    img_stretched = clahe.apply(img_stretched)

# 边缘保持滤波 - 双边滤波
img_filtered = cv2.bilateralFilter(img_stretched, d=9, sigmaColor=75, sigmaSpace=75)

# 锐化滤波 - Unsharp Masking
gaussian = cv2.GaussianBlur(img_filtered, (9, 9), 10.0)
img_sharp = cv2.addWeighted(img_filtered, 1.5, gaussian, -0.5, 0, img_filtered)

# 保存优化后的图像
cv2.imwrite("optimized_imagea.png", img_sharp)


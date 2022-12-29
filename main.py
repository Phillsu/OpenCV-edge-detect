import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 簽名檔
def add_signature(src, pic):
    h = src.shape[0]
    w = src.shape[1]
    h1 = pic.shape[0]
    w1 = pic.shape[1]
    dst = np.zeros((h, w))
    for item in range(h):
        for item2 in range(w):
            if src[item, item2] == 0:
                dst[item, item2] = 1
                cv.circle(pic, (item2 * 3, item * 3), 5, (255, 0, 0), -1)
    return pic

# 點
def point(x, i, j):
    #這裡會爆一個zero div... 的錯 後來讓i != 0 就解決了
    if i ==0:
        i = 0.001
    y = -x * math.cos(i * math.pi / 180) / math.sin(i * math.pi / 180) + (j - max_rho / 2) / math.sin(i * math.pi / 180)
    return y


img = 'test1.jpg'
img = cv.imread(img, cv.IMREAD_GRAYSCALE)
name = cv.imread("name2.png")
name = cv.cvtColor(name, cv.COLOR_RGB2GRAY)

# Gaussian Blur 這裡應該可以用function 嗎...
img = cv.GaussianBlur(img, (3, 3), 0)

# Sobel filter -1 這裡應該可以用function  嗎...
x = cv.Sobel(img, cv.CV_16S, 1, 0)
y = cv.Sobel(img, cv.CV_16S, 0, 1)

# Sobel filter -2 轉回uint8
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)

M = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

w, h = img.shape

# math.atan2 =是計算弧度 角度 = 弧度*180/ pi
Xta = np.zeros([w, h])
for i in range(w):
    for j in range(h):
        Xta[i, j] = math.atan2(absY[i, j], absX[i, j]) * 180 / math.pi

# Non-Maximum Suppression
# 將Xta劃分到4個區域 0,90,45,-45
for i in range(w):
    for j in range(h):
        if (Xta[i, j] > -22.5 and Xta[i, j] <= 22.5) or (Xta[i, j] > 157.5 and Xta[i, j] <= -157.5):
            Xta[i, j] = 0
        elif (Xta[i, j] > 22.5 and Xta[i, j] <= 67.5) or (Xta[i, j] > -157.5 and Xta[i, j] <= -112.5):
            Xta[i, j] = -45
        elif (Xta[i, j] > 67.5 and Xta[i, j] <= 112.5) or (Xta[i, j] > -112.5 and Xta[i, j] <= -67.5):
            Xta[i, j] = 90
        else:
            Xta[i, j] = 45

new_img = np.zeros([w, h])

# 遍歷3*3區域 保留連續點最大值
for i in range(2, (w - 1)):
    for j in range(2, (h - 1)):
        if (Xta[i, j] == 0 and M[i, j] == max(M[i, j], M[i, j + 1], M[i, j - 1])):
            new_img[i, j] = M[i, j]
        elif (Xta[i, j] == -45 and M[i, j] == max(M[i, j], M[i + 1, j - 1], M[i - 1, j + 1])):
            new_img[i, j] = M[i, j]
        elif (Xta[i, j] == 90 and M[i, j] == max(M[i, j], M[i + 1, j], M[i - 1, j])):
            new_img[i, j] = M[i, j]
        elif (Xta[i, j] == 45 and M[i, j] == max(M[i, j], M[i + 1, j + 1], M[i - 1, j - 1])):
            new_img[i, j] = M[i, j]

# 雙門檻值測跟邊緣連接
T_H = 0.2 * np.max(new_img)
T_L = 0.1 * np.max(new_img)
canny_img = np.zeros([w, h])
for i in range(w):
    for j in range(h):
        if (new_img[i, j] > T_H):
            canny_img[i, j] = 1

        elif (new_img[i, j] > T_L and new_img[i, j] < T_H):
            for local_w in range(i - 1, i + 1):
                for local_h in range(j - 1, j + 1):
                    if (local_w != i and local_h != j) and new_img[local_w, local_h] > T_H:
                        canny_img[i, j] = 1

# plot預設會自動補色 用cmap指定為grayscale
# canny_img = add_signature(name,canny_img)
plt.imshow(canny_img, cmap='gray')
plt.show()

# hough transform
max_rho = round(2 * math.sqrt(w ** 2 + h ** 2))
hough_space = np.zeros([180, max_rho])

for i in range(w):
    for j in range(h):
        if canny_img[i, j] == 1:
            for theta in range(180):
                rho = i * math.cos(theta * math.pi / 180) + j * math.sin(theta * math.pi / 180)
                rho = round(rho + max_rho / 2)

                hough_space[theta, rho] += 1
                hough_space[theta, rho + 1] += 1
                hough_space[theta, rho - 1] += 1

# draw line
T = 0.5 * np.max(hough_space)
for i in range(180):
    for j in range(max_rho):
        if hough_space[i, j] >= T:
            x = [1, w]
            y = [point(1, i, j), point(w, i, j)]
            plt.plot(y, x, color="white", linewidth=1)

img = add_signature(name, img)
plt.imshow(img, cmap='gray')
plt.show()

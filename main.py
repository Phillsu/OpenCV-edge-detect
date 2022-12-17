import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img1 = 'test1.jpg'
img = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (5, 5), 0.5)
cv2.imshow(img)
'''
w, h = img.shape

# 計算梯度的幅值圖像和角度圖像
Px = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype='float32')
Py = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]), dtype='float32')

gx = cv.filter2D(img, -1, Px)
gy = cv.filter2D(img, -1, Py)

M = abs(gx) + abs(gy)
Xta = np.zeros([w, h])

for i in range(w):
    for j in range(h):
        Xta[i, j] = math.atan2(gy[i, j], gx[i, j]) * 180 / math.pi

# 非極大值抑制
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

# 遍歷3*3區域
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

# plt.imshow(M)
# plt.imshow(new_img)
# plt.show()

# 雙閾值檢測和邊緣連接
T_H = 0.9 * np.max(new_img)
T_L = 0.3 * np.max(new_img)
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

# plt.imshow(canny_img)
# plt.show()

# 霍夫變換
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

plt.imshow(cv.resize(hough_space, (w, h)))
plt.show()


def point(x, i, j):
    if i == 0:
        i = 0.01
    y = -x * math.cos(i * math.pi / 180) / math.sin(i * math.pi / 180) + (j - max_rho / 2) / math.sin(i * math.pi / 180)

    return y


# 直線偵測
T = 0.5 * np.max(hough_space)
for i in range(180):
    for j in range(max_rho):
        if hough_space[i, j] >= T:
            x = [1, w]
            y = [point(1, i, j), point(w, i, j)]
            plt.plot(y, x, color="red", linewidth=1)

plt.imshow(img)
plt.show()

'''
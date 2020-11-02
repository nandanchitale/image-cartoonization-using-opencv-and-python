import cv2
import numpy as np

num_down = 1        # Number of downsampling steps
num_bilateral = 500   # Number of bilateral filtering steps

img_rgb = cv2.imread("pp.jfif")

img_rgb = cv2.resize(img_rgb, (800,800))

img_color = img_rgb
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)

for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(
        img_color,
        d = 9,
        sigmaSpace = 7,
        sigmaColor = 8
    )

for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

img_edge = cv2.adaptiveThreshold(
    img_blur, 
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize = 5,
    C = 4
)

img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
img_cartoon = cv2.bitwise_and(img_color, img_edge)

cv2.imwrite('cartoon.png',img_cartoon)
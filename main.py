import cv2
import numpy as np

GAMMA_CORRECTION = float(1 / 3) # 단계 3에서 필요한 상수입니다.
orig_img_path = ['./orig_img/img_1.png', './orig_img/img_2.png']
orig_img_RGB = [cv2.imread(orig_img_path[0]), cv2.imread(orig_img_path[1])]
height, width, _ = orig_img_RGB[0].shape

## 단계 1. RGB를 YUV로 변환합니다.
orig_img_YUV_cv = []
orig_img_YUV = []
for i in range(2):
    # 논문의 방식을 따르지 않고 OpenCV 자체에 내장된 메소드를 활용하여 YUV 이미지를 생성합니다.
    orig_img_YUV_cv.append(cv2.cvtColor(orig_img_RGB[i], cv2.COLOR_RGB2YUV))
    filename = './orig_img/cv_' + str(i + 1) + '.png'
    cv2.imwrite(filename, orig_img_YUV_cv[i])
    # 논문의 방식대로 행렬의 곱을 활용해 직접 YUV 이미지를 생성합니다.
    tmp = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            tmp[h, w, 0] = (0.299 * orig_img_RGB[i][h, w, 0]) + (0.587 * orig_img_RGB[i][h, w, 1]) + (0.114 * orig_img_RGB[i][h, w, 2])
            tmp[h, w, 1] = (-0.169 * orig_img_RGB[i][h, w, 0]) + (-0.331 * orig_img_RGB[i][h, w, 1]) + (0.500 * orig_img_RGB[i][h, w, 2])
            tmp[h, w, 2] = (0.500 * orig_img_RGB[i][h, w, 0]) + (-0.419 * orig_img_RGB[i][h, w, 1]) + (-0.081 * orig_img_RGB[i][h, w, 2])
    orig_img_YUV.append(tmp)
    filename = './orig_img/ch_' + str(i + 1) + '.png'
    cv2.imwrite(filename, orig_img_YUV[i])

## 단계 2. 이미지가 저조도 이미지인지 판단합니다.
is_it_dark_cv = []
is_it_dark = []
for i in range(2):
    Y_sum_cv = 0
    Y_sum = 0
    for h in range(height):
        for w in range(width):
            Y_sum_cv += orig_img_YUV_cv[i][h, w, 0]    # OpenCV로 생성한 YUV 이미지
            Y_sum += orig_img_YUV[i][h, w, 0]          # 직접 생성한 YUV 이미지
    Y_mean_cv = Y_sum_cv / (height * width)
    is_it_dark_cv.append(bool(Y_mean_cv < 40.0))
    Y_mean = Y_sum / (height * width)
    is_it_dark.append(bool(Y_mean < 40.0))

## 단계 3. 저조도로 판단된 모든 이미지(RGB)에 대해 Gamma Correction을 적용합니다.
dstimg = []
lookupTable = []
for i in range(256):
    lookupTable.append((i / 255) ** GAMMA_CORRECTION)
for i in range(2):
    tmp = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            for col in range(3):
                LUT = lookupTable[int(orig_img_RGB[i][h, w, col])]
                tmp[h, w, col] = LUT * 255 + 0.5
                tmp[h, w, col] = int(min(max(tmp[h, w, col], 0.0), 255))
    # dstimg.append(np.power(orig_img_RGB[i] / 255.0, GAMMA_CORRECTION) * 255.0)
    dstimg.append(tmp)
    filename = './gamma_img/ch_' + str(i + 1) + '.png'
    cv2.imwrite(filename, dstimg[i])

## 단계 4. (저조도 이미지만) Edge를 제외한 부분의 노이즈 제거를 위해 Bilateral Filter를 적용합니다.
bt_img = []
for i in range(2):
    bt_img.append(cv2.bilateralFilter(dstimg[i], -1, 10, 5))
    filename = './bilateral_img/bt_' + str(i + 1) + '.png'
    cv2.imwrite(filename, bt_img[i])

## 단계 5. 모든 이미지를 선명하게 만들기 위해 Image Sharpening Filter를 적용합니다.
shp_img = []
sharpening_kernel = np.array([[-1, -1, -1, -1, -1],
                              [-1, 2, 2, 2, -1],
                              [-1, 2, 8, 2, -1],
                              [-1, 2, 2, 2, -1],
                              [-1, -1, -1, -1, -1]]) / 8.0
for i in range(2):
    shp_img.append(cv2.filter2D(bt_img[i], -1, sharpening_kernel))
    filename = './sharpened_img/shp_' + str(i + 1) + '.png'
    cv2.imwrite(filename, shp_img[i])
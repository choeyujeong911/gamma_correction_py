import cv2
import numpy as np

NUM_OF_IMG = int(3)
GAMMA_CORRECTION = float(1 / 3) # 단계 3에서 필요한 상수입니다.
# 실습을 위한 원본 이미지는 ./practice/0__orig_img/ 디렉토리 내에 있어야 합니다. 실습 이미지의 이름에 맞게 경로 수정 바랍니다.
orig_img_path = ['./practice/0__orig_img/img_1.png', './practice/0__orig_img/img_2.png', './practice/0__orig_img/img_3.png']
orig_img_RGB = [cv2.imread(orig_img_path[0]), cv2.imread(orig_img_path[1]), cv2.imread(orig_img_path[2])]
height, width, _ = orig_img_RGB[0].shape

## 단계 1. RGB를 YUV로 변환합니다.
orig_img_YUV = []
for i in range(NUM_OF_IMG):
    # 행렬의 곱을 활용해 직접 YUV 이미지를 생성합니다.
    tmp = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            tmp[h, w, 0] = (0.299 * orig_img_RGB[i][h, w, 0]) + (0.587 * orig_img_RGB[i][h, w, 1]) + (0.114 * orig_img_RGB[i][h, w, 2])
            tmp[h, w, 1] = (-0.169 * orig_img_RGB[i][h, w, 0]) + (-0.331 * orig_img_RGB[i][h, w, 1]) + (0.500 * orig_img_RGB[i][h, w, 2])
            tmp[h, w, 2] = (0.500 * orig_img_RGB[i][h, w, 0]) + (-0.419 * orig_img_RGB[i][h, w, 1]) + (-0.081 * orig_img_RGB[i][h, w, 2])
    orig_img_YUV.append(tmp)
    filename = './practice/1__YUV_img/yuv_' + str(i + 1) + '.png'
    cv2.imwrite(filename, orig_img_YUV[i])
print("> > 1단계(RGB-YUV 변환) 완료")
print()

## 단계 2. 이미지가 저조도 이미지인지 판단합니다.
is_it_dark = []
for i in range(NUM_OF_IMG):
    # Y_sum_cv = 0
    Y_sum = 0
    for h in range(height):
        for w in range(width):
            Y_sum += orig_img_YUV[i][h, w, 0]
    Y_mean = Y_sum / (height * width)
    print("* img_{0}.png의 Y_mean: {1:.2f}".format(i + 1, Y_mean))
    is_it_dark.append(bool(Y_mean < 40.0))
    print("> 저조도 이미지 ? : {0}".format(is_it_dark[i]))
print("> > 2단계(저조도 이미지 판단) 완료")
print()

## 단계 3. 저조도로 판단된 모든 이미지(RGB)에 대해 Gamma Correction을 적용합니다.
lookupTable = []
for i in range(256):    # 룩업테이블(상수 리스트)을 정의합니다.
    lookupTable.append((i / 255) ** GAMMA_CORRECTION)
dstimg = []
for i in range(NUM_OF_IMG): # 룩업테이블을 기반으로 Gamma Correction을 적용합니다.
    if is_it_dark[i]:
        tmp = np.zeros((height, width, 3), dtype=np.uint8)
        for h in range(height):
            for w in range(width):
                for col in range(3):
                    LUT = lookupTable[int(orig_img_RGB[i][h, w, col])]
                    tmp[h, w, col] = LUT * 255 + 0.5
                    tmp[h, w, col] = int(min(max(tmp[h, w, col], 0.0), 255))
        # dstimg.append(np.power(orig_img_RGB[i] / 255.0, GAMMA_CORRECTION) * 255.0)
        dstimg.append(tmp)
        filename = './practice/3__gamma_img/gm_' + str(i + 1) + '.png'
        cv2.imwrite(filename, dstimg[i])
print("> > 3단계(저조도 이미지의 감마 보정) 완료")
print()

## 단계 4. (저조도 이미지만) Edge를 제외한 부분의 노이즈 제거를 위해 Bilateral Filter를 적용합니다.
bt_img = []
for i in range(NUM_OF_IMG):
    if is_it_dark[i]:
        bt_img.append(cv2.bilateralFilter(dstimg[i], -1, 10, 5))
        filename = './practice/4__bilateral_img/bt_' + str(i + 1) + '.png'
        cv2.imwrite(filename, bt_img[i])
    else:
        bt_img.append(orig_img_RGB[i])
print("> > 4단계(저조도 이미지의 노이즈 제거) 완료")
print()

## 단계 5. 모든 이미지를 선명하게 만들기 위해 Image Sharpening Filter를 적용합니다.
shp_img = []
sharpening_kernel = np.array([[-1, -1, -1, -1, -1],
                              [-1, 2, 2, 2, -1],
                              [-1, 2, 8, 2, -1],
                              [-1, 2, 2, 2, -1],
                              [-1, -1, -1, -1, -1]]) / 8.0
for i in range(NUM_OF_IMG):
    if is_it_dark[i]:   # 저조도 이미지의 경우, 감마 보정 후 노이즈 제거까지 완료된 이미지를 사용합니다.
        shp_img.append(cv2.filter2D(bt_img[i], -1, sharpening_kernel))
    else:               # 일반 이미지의 경우, 원본(RGB) 그대로 사용합니다.
        shp_img.append(cv2.filter2D(orig_img_RGB[i], -1, sharpening_kernel))
    filename = './practice/5__sharpened_img/shp_' + str(i + 1) + '.png'
    cv2.imwrite(filename, shp_img[i])
print("> > 5단계(모든 이미지를 선명하게 보정) 완료")
print()

print("** 모든 작업이 완료되었습니다. **")
print("** ./practice/ 디렉토리에서 각 단계의 이미지를 확인할 수 있습니다. **")
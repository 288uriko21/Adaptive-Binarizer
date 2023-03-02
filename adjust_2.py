from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
import adaptive_binarizer as ab
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import math

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tessdata"'

image = cv2.imread("dataset/test/2.png")

block_size = max(image.shape[1], image.shape[0])
delta = 40
cropInd = 0.02
deltaCrop = 30

while True:
    cropped = np.median(image[0:image.shape[0], 0:round(image.shape[1] * cropInd)])
    cropped += np.median(image[0:image.shape[0], round(image.shape[1] * (1 - cropInd)):image.shape[1]])
    cropped += np.median(image[0:round(image.shape[0] * cropInd), 0:image.shape[1]])
    cropped += np.median(image[round(image.shape[0] * (1 - cropInd)):image.shape[0], 0:image.shape[1]])
    cropped //= 4

    if (cropped < np.median(image) - deltaCrop):
        image = image[round(image.shape[0] * cropInd):round(image.shape[0] * (1 - cropInd)),
                round(image.shape[1] * cropInd):round(image.shape[1] * (1 - cropInd))]
    else:
        break

result = Image.fromarray(image)
result.save('dataset/test/3.png')

SubMean = 20
SubBlock = 10

NumOfBlock = 0
BlockMedianSum = 0
while True:
    obj = ab.AdaptiveBinarizer(block_size, delta)
    flag = 0
    NumOfBlock = 0
    BlockMedianSum = 0

    for row in range(0, image.shape[0], obj.block_size):
        for col in range(0, image.shape[1], obj.block_size):
            idx = (row, col)
            block_idx = obj.get_block_index(image.shape, idx)
            elem = image[tuple(block_idx)]
            NumOfBlock += 1
            BlockMedianSum += np.median(elem)

            top_right = np.mean(elem[round(0.5 * elem.shape[0]) : elem.shape[0], 0 : round(0.5 * elem.shape[1])])
            top_left = np.mean(elem[0 : round(0.5 * elem.shape[0]), 0 : round(0.5 * elem.shape[1])])
            low_right = np.mean(elem[round(0.5 * elem.shape[0]) : elem.shape[0], round(0.5 * elem.shape[1]) : elem.shape[1]])
            low_left = np.mean(elem[0 : round(0.5 * elem.shape[0]), round(0.5 * elem.shape[1]) : elem.shape[1]])

            if abs(top_right - top_left) > SubMean:
                block_size -= SubBlock
                flag = 1
            if (abs(top_right - low_right) > SubMean):
                block_size -= SubBlock
                flag = 1
            if (abs(top_right - low_left) > SubMean):
                block_size -= SubBlock
                flag = 1
            if (abs(top_left - low_right) > SubMean):
                block_size -= SubBlock
                flag = 1
            if (abs(top_left - low_left) > SubMean):
                block_size -= SubBlock
                flag = 1
            if (abs(low_right - low_left) > SubMean):
                block_size -= SubBlock
                flag = 1

            if flag:
                break
        if flag:
            break
    if not(flag):
        break
    if block_size < 0:
        block_size = 40
        NumOfBlock = 0
        BlockMedianSum = 0
        break

print(block_size)

image = ab.preprocess(image)
for row in range(0, image.shape[0], obj.block_size): # возможно, лучше по всей картинке что-то смотреть
    for col in range(0, image.shape[1], obj.block_size):
        idx = (row, col)
        block_idx = obj.get_block_index(image.shape, idx)
        elem = image[tuple(block_idx)]
        NumOfBlock += 1
        BlockMedianSum += np.median(elem)

GlobalMed = BlockMedianSum / NumOfBlock

# теперь надо как-то аппроксимировать f(GlobalMed) = delta

obj1 = ab.AdaptiveBinarizer(40, 5)


# obj1 = ab.AdaptiveBinarizer(40, 5)
#
# img = obj1.binarize(image)
# result = Image.fromarray(image)
# result.save('dataset/0.png')
#
# text = pytesseract.image_to_string(img, lang="rus", config="--psm 6")
#
# fi = open('dataset/test/1.txt', 'r', encoding="utf8")
# ideal = fi.read()
# fi.close()
#
# a = fuzz.ratio(text, ideal)
# print("Accuracy binarized:", a)

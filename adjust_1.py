from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
import adaptive_binarizer as ab
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt

DELTA = 30

image = cv2.imread("dataset/test/2.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = Image.fromarray(image)
result.save('dataset/test/3.png')

medIm = np.median(image)

pred = 10
n = 10
left = 0
for i in range(20, image.shape[1], 10):
    cropped_image = image[0:image.shape[0], pred:i]
    if(np.median(cropped_image) > medIm-DELTA):
        left = i
        if (left==20):
            left = 0
        break
    pred = i
    n += 1

pred = 10
top = 0
n = 10
for i in range(20, image.shape[0], 10):
    cropped_image = image[pred:i, 0:image.shape[1]]
    if(np.median(cropped_image) > medIm-DELTA):
       # print(i)
        top = i
        if (top == 20):
            top = 0
        break
    pred = i
    n += 1


pred = image.shape[1] - 10
right = 0
n = 10
for i in reversed(range(20, image.shape[1]-10, 10)):
    cropped_image = image[0:image.shape[0], i:pred]
    if(np.median(cropped_image) > medIm-DELTA):
        #print(i)
        right = i
        if (right == image.shape[0]-10):
            right = image.shape[0]
        break
    pred = i
    n += 1

pred = image.shape[0] - 10
low = 0
n = 10
for i in reversed(range(20, image.shape[0]-10, 10)):
    cropped_image = image[i:pred, 0:image.shape[1]]
    if(np.median(cropped_image) > medIm-DELTA):
        #print(i)
        low = i
        if (low == image.shape[1]-10):
            low = image.shape[0]
        break
    pred = i
    n += 1

image = image[top:low, left:right]
result = Image.fromarray(image)
result.save('dataset/test/4.png')


block_size = max(image.shape[1], image.shape[0])
delta = round(((14400)/7) / (medIm + (20/7)))

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tessdata"'

obj1 = ab.AdaptiveBinarizer(block_size, delta)
img = cv2.imread("dataset/test/4.png")
img2 = img
img = obj1.binarize(img)
result = Image.fromarray(img)
result.save('dataset/test/0.png')

text = pytesseract.image_to_string(img, lang="rus", config="--psm 6")

text2 = pytesseract.image_to_string(img2, lang="rus", config="--psm 6")

fi = open('dataset/test/1.txt', 'r', encoding="utf8")
ideal = fi.read()
fi.close()


a = fuzz.ratio(text, ideal)
b = fuzz.ratio(text2, ideal)
print("Accuracy binarized:", a)
print("Accuracy NOT binarized:", b)
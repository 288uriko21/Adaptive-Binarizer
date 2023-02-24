from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
import adaptive_binarizer as ab
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import random

image = cv2.imread("dataset/text13/18.png")
#src = cv2.imread("img.png")
sr = np.zeros_like(image)
bl, g, r = cv2.split(sr)

a = random.randint(3, 10)
b = random.randint(220, 250)

obj = ab.AdaptiveBinarizer(900, 70)

for row in range(0, bl.shape[0], 900):
   for col in range(0, bl.shape[1], 900):
        block_idx = obj.get_block_index(bl.shape, (row, col))
        bl[tuple(block_idx)] = bl[tuple(block_idx)] - random.randint(0, 255)
        break
   break


# for row in range(0, g.shape[0], a):
#    for col in range(0, g.shape[1], b):
#         block_idx = obj.get_block_index(g.shape, (row, col))
#         g[tuple(block_idx)] = g[tuple(block_idx)] - random.randint(0, 255)
#
# for row in range(0, r.shape[0], a):
#     for col in range(0, r.shape[1], b):
#         block_idx = obj.get_block_index(r.shape, (row, col))
#         r[tuple(block_idx)] = r[tuple(block_idx)] - random.randint(0, 255)

imageback = cv2.merge([r, g, bl])
#print(r)
result = Image.fromarray(imageback)
result.save('dataset/text11/00.png')

# i = 0
# n = 1
# s = 'dataset/text13/'
# while (i < 0.9):
#     out_image = cv2.addWeighted(image, 1 - i, imageback, i, 0.0)
#     result = Image.fromarray(out_image)
#     s1 = s + str(n) + '.png'
#     result.save(s1)
#     n += 1
#     i += 0.03













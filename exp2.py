from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
import adaptive_binarizer as ab
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt



pytesseract.pytesseract.tesseract_cmd = r'C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tessdata"'

fi = open('dataset/text19/1.txt', 'r', encoding="utf8")
ideal = fi.read()
fi.close()

# s = "dataset/text7/"
# f = open('dataset/text7_2/2.txt', 'w')
Uvalues = [[], []]



for delta in range(35, 46, 2):
    Uvalues[0].clear()
    Uvalues[1].clear()
    for block_size in range(25, 900, 40):
        img = cv2.imread('dataset/text19/2.png')
        obj1 = ab.AdaptiveBinarizer(block_size, delta)
        img = obj1.binarize(img)
        text = pytesseract.image_to_string(img, lang="rus", config="--psm 6")
        a = fuzz.ratio(text, ideal)
        Uvalues[0].append(block_size)
        Uvalues[1].append(a)
    # f.write('accuracy ' + str(a) + ' ; ' + 'block_size ' + str(block_size) + '\n')
    plt.plot(Uvalues[0], Uvalues[1], 'ro')
    plt.savefig("dataset/text19/Delta_" + str(delta) + ".png")
    plt.close()
    #plt.show()
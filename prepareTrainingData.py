import cv2
import pytesseract
import adaptive_binarizer as ab
from fuzzywuzzy import fuzz
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tessdata"'

f = open('traindata/2.txt', 'w')
folder = 1
for i in [27, 20, 13, 13, 19, 19, 21, 20, 20, 19, 11, 2]:
    fi = open('traindata/train' + str(folder) + '/1.txt', 'r', encoding="utf8")
    ideal = fi.read()
    fi.close()
    for pic in range(2, i+1):
        image = cv2.imread('traindata/train' + str(folder) + '/' + str(pic) + '.png')
        block_size = max(image.shape[1], image.shape[0])
        maxAcur = 0
        bestDelta = 0
        for delta in range(0, 170, 5):
            img = image
            obj = ab.AdaptiveBinarizer(block_size, delta)
            img = obj.binarize(img)
            text = pytesseract.image_to_string(img, lang="rus", config="--psm 6")
            a = fuzz.ratio(text, ideal)
            if (a > maxAcur):
                maxAcur = a
                bestDelta = delta
        bl, g, r = cv2.split(image)
        cropped = np.median(image[0:image.shape[0], 0:round(image.shape[1] * 0.08)])
        cropped += np.median(image[0:image.shape[0], round(image.shape[1] * (1 - 0.08)):image.shape[1]])
        cropped += np.median(image[0:round(image.shape[0]*0.08), 0:image.shape[1]])
        cropped += np.median(image[round(image.shape[0] * 0.92):image.shape[0], 0:image.shape[1]])
        cropped //= 4
        f.write(str(pic) + ' ' + str(np.median(bl)) + ' ' + str(np.median(g)) + ' ' + str(np.median(r)) +
                '  ' + str(bestDelta) + '  ' + str(maxAcur) + ' ' +
                str(round(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))) + ' ' + str(cropped) + '\n')
    folder += 1
f.close()


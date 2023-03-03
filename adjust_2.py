import cv2
from PIL import Image
import numpy as np
import pytesseract
import adaptive_binarizer as ab
from fuzzywuzzy import fuzz

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "C:\Users\iripa\AppData\Local\Programs\Tesseract-OCR\tessdata"'


def binarizeProccess(s, s1):
    image = cv2.imread(s)  # целевая картинка
    imgcopy = image
    block_size = max(image.shape[1], image.shape[0])
    delta = 40
    cropInd = 0.02
    deltaCrop = 20

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

    # result = Image.fromarray(image)
    # result.save('traindata/train4/1000.png')

    SubMean = 7
    SubBlock = 10

    BS = block_size

    while True:
        obj = ab.AdaptiveBinarizer(BS, delta)
        flag = 0

        for row in range(0, image.shape[0], obj.block_size):
            for col in range(0, image.shape[1], obj.block_size):
                idx = (row, col)
                block_idx = obj.get_block_index(image.shape, idx)
                elem = image[tuple(block_idx)]

                top_right = np.mean(elem[round(0.5 * elem.shape[0]): elem.shape[0], 0: round(0.5 * elem.shape[1])])
                top_left = np.mean(elem[0: round(0.5 * elem.shape[0]), 0: round(0.5 * elem.shape[1])])
                low_right = np.mean(
                    elem[round(0.5 * elem.shape[0]): elem.shape[0], round(0.5 * elem.shape[1]): elem.shape[1]])
                low_left = np.mean(elem[0: round(0.5 * elem.shape[0]), round(0.5 * elem.shape[1]): elem.shape[1]])

                # low_left_mini = np.mean(elem[0 : round(0.25 * elem.shape[0]), round(0.25 * elem.shape[1]) : elem.shape[1]])
                # top_right_mini = np.mean(elem[round(0.25 * elem.shape[0]) : elem.shape[0], 0 : round(0.25 * elem.shape[1])])

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
                # if abs(top_right_mini - low_left_mini) > SubMean:
                #     block_size -= SubBlock
                #     flag = 1

                if flag:
                    break
            if flag:
                break
        if not (flag):
            break
        if block_size < 0:
            block_size = 40
            break

    # print(block_size)
    # result = Image.fromarray(elem)
    # result.save('traindata/train4/1001.png')

    obj1 = ab.AdaptiveBinarizer(40, 5)
    img = obj1.preprocess(image)
    maxMed = 0
    minMed = 255
    for row in range(0, img.shape[0], obj.block_size):
        for col in range(0, img.shape[1], obj.block_size):
            idx = (row, col)
            block_idx = obj.get_block_index(img.shape, idx)
            elem = img[tuple(block_idx)]
            a = np.median(elem)
            if a < minMed:
                minMed = a
            if a > maxMed:
                maxMed = a

    maxPartText = 0.17

    delta = (maxMed - minMed)*10

    r = np.sum(img > -1)
    p = np.sum((img - ((maxMed + minMed) // 2)) < delta)

    # print(delta)

    while (np.sum((img - ((maxMed + minMed) // 2)) > delta)) > (np.sum(img > -1) * maxPartText):
        delta += 1

    # print(delta)

    if delta < 0:
        delta = 0

    obj1 = ab.AdaptiveBinarizer(block_size, delta)
    obj = ab.AdaptiveBinarizer(40, 25)

    img = obj1.binarize(image)
    img2 = obj.binarize(imgcopy)

    text = pytesseract.image_to_string(img, lang="rus", config="--psm 6")
    text2 = pytesseract.image_to_string(img2, lang="rus", config="--psm 6")

    fi = open(s1, 'r', encoding="utf8")  # файл с текстом, который реально на картинке
    ideal = fi.read()
    fi.close()
    lst = []
    a = fuzz.ratio(text, ideal)
    b = fuzz.ratio(text2, ideal)
    lst.append(a)
    lst.append(b)
    print(a, b)
    return lst

S = 0
S1 = 0
for i in range(1, 31):
    s = 'testingset/' + str(i) + '.png'
    s1 = 'testingset/' + str(i) + '.txt'
    lst = binarizeProccess(s, s1)
    S += lst[0]
    S1 += lst[1]

print("Glabal: ", S/30, S1/30)





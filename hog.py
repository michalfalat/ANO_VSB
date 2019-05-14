
# import matplotlib.pyplot as plt
import numpy as np
import math

import cv2


class HogDecsriptor:
    def __init__(self, name, age):
        self.name = name
        self.age = age


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow2(img):
    npimg = img
    plt.imshow(npimg)
    plt.show()


def getPixelBrightness(image, x, y):
    vals = image[x, y]
    # if vals > 0:
    #     print('valuddde:')
    #     print(vals)
    return int(vals)


def getFx(image, x, y):
    b1 = getPixelBrightness(image, x+1, y)
    b2 = getPixelBrightness(image, x, y)
    return b1 - b2


def getFy(image, x, y):
    res = getPixelBrightness(image, x, y+1) - getPixelBrightness(image, x, y)
    return res


def calcOrientation(fx, fy):
    if fx == 0:
        return math.degrees(np.pi/2) + 90
    else:
        return math.degrees(np.arctan(fy/fx)) + 90


def calcGradient(fx, fy):
    return np.sqrt((np.power(fx, 2) + np.power(fy, 2)))


def normalise(data):
    dataSum = sum(data)
    if dataSum == 0:
        return data
    norm = [float(i)/sum(data) * 10 for i in data]
    return norm


def drawArrows(image, x, y, vector):
    for i in range(9):
        angle = 40 * i
        length = int(vector[i])
        x2 = int(x + length * math.cos(angle * 3.14 / 180.0))
        y2 = int(y + length * math.sin(angle * 3.14 / 180.0))
        cv2.arrowedLine(image, (x, y), (x2, y2), (255, 0, 0), 1)
    return image


def neural_network(features, labels):
    nclasses = labels.cols
    nfeatures = features.cols
    ann = cv2.ml.ANN_MLP.create()
    Mat_ < int > layers(3, 1)
    layers[0, 0] = nfeatures
    layers[1, 0] = 6
    layers[2, 0] = nclasses
    ann.setLayerSizes(layers)
    ann.setActivationFunction(cv2.SIGMOID_SYM)
    ann.setTermCriteria(1000000)
    ann.setTrainMethod(cv2.BACKPROP, 0.0001)

    print("Training...\n")
    ann.train(features, labels)

    result = None
    pred = ann.predict(features.row(0), result)


def main():

    # train(trainloader, model)
    # validation(testloader, model)

    img = cv2.imread('test.png', 0)
    height, width = img.shape
    BLOCK_SIZE = 2
    CELL_SIZE = 8
    BINS = 9

    # brightness = np.zeros((height,width))

    orientation = np.zeros((height, width))
    gradient = np.zeros((height, width))

    for h in range(height-1):
        for w in range(width-1):
            # brightness[h][w] =  getPixelBrightness(img, h,w)
            fx = getFx(img, h, w)
            fy = getFy(img, h, w)

            # print(fy)
            orientation[h][w] = calcOrientation(fx, fy)
            # print(orientation[h][w])
            gradient[h][w] = calcGradient(fx, fy)
        pass


#
#
    cellsInHeight = int(math.ceil(height/CELL_SIZE))
    cellsInWidth = int(math.ceil(width/CELL_SIZE))

    # cellsGradient = np.zeros((cellsInHeight, cellsInWidth)
    # cellsOrientation = np.zeros(height/CELL_SIZE, width/CELL_SIZE)
    # blocks = np.zeros(height/BLOCK_SIZE, width/BLOCK_SIZE)
    bNumber = 0
    allVectors = []
    rows, cols = (cellsInHeight, cellsInWidth)

    arrayOfVectors = [[0]*cols]*rows

    for ch in range(cellsInHeight):
        for cw in range(cellsInWidth):
            vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for h in range(CELL_SIZE):
                for w in range(CELL_SIZE):
                    imgX = (ch * CELL_SIZE) + h
                    imgY = (cw * CELL_SIZE) + w
                    if(imgX >= height):
                        imgX = height-1
                    if(imgY >= width):
                        imgY = width-1
                    # cellsGradient[h % CELL_SIZE][w % CELL_SIZE] += gradient[h][w]
                    # cellsGradient[h % CELL_SIZE][w % CELL_SIZE] += gradient[h][w]
                    # TODO ONE BLOCK
                    # if ch == 0 and cw == 2:
                    #print(str(imgX) + " " + str(imgY) + ": " + str(orientation[imgX][imgY]) + "\t " + str(gradient[imgX][imgY]))
                    binN = int(orientation[imgX][imgY]/20)
                    if binN > 8:
                        binN = 8
                        # print(str(binN) + " - " + str(orientation[imgX][imgY]))
                    vector[binN] += gradient[imgX][imgY]
            allVectors.append(vector)
            arrayOfVectors[ch][cw] = vector
            # cell is done
            norm = normalise(vector)
            xCenter = int(ch * CELL_SIZE + CELL_SIZE/2)
            yCenter = int(cw * CELL_SIZE + CELL_SIZE/2)
            #  print("\n\n")
            # print(vector)
            # print(norm)

    # print("FINISH")
    vectorCounter = 0

    arrayOfBlocks = []
    for r in range(0, (rows), BLOCK_SIZE):  # 0 - 7
        for c in range(0, (cols), BLOCK_SIZE):  # 0 - 7

            tmpVector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for his in range(9):
                tmpVector[his] += arrayOfVectors[r][c][his]
                tmpVector[his] += arrayOfVectors[r][c+1][his]
                tmpVector[his] += arrayOfVectors[r+1][c][his]
                tmpVector[his] += arrayOfVectors[r+1][c+1][his]
            print("Vector: ")
            print(tmpVector)

            norm = normalise(tmpVector)
            rowPos = int((CELL_SIZE) * r + (CELL_SIZE * BLOCK_SIZE) / 2)
            cellPos = int((CELL_SIZE) * c + (CELL_SIZE * BLOCK_SIZE) / 2)
            img = drawArrows(img, rowPos, cellPos, norm)
            print("Normalised vector x 10: ")
            print(norm)
            print("\n")
            vectorCounter += 1
    cv2.imshow('Data', img)
    #cv2.namedWindow('Data', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Data', 600, 600)

    cv2.imwrite("square_circle_opencv.jpg", img)
    cv2.waitKey()


if __name__ == '__main__':
    main()

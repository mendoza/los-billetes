import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pytesseract


def biggestRectangle(contours):
    biggest = None
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
        i = contours[index]
        area = cv.contourArea(i)
        if area > 100:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.1*peri, True)
            if area > max_area:  # and len(approx)==4:
                biggest = approx
                max_area = area
                indexReturn = index
    return indexReturn


def calculateFeatures(path):
    img = cv.imread(path)  # you can use any image you want.
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(img)

    negative = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    plt.subplot(2, 3, 2)
    plt.title("B&W")
    plt.imshow(negative, 'gray')

    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    sobel = (grad * 255 / grad.max()).astype('uint8')
    plt.subplot(2, 3, 3)
    plt.title("Sobel")
    plt.imshow(sobel, 'gray')

    gaussian = cv.GaussianBlur(sobel, (3, 3), 0).astype('uint8')
    plt.subplot(2, 3, 4)
    plt.title("Gaussian blur")
    plt.imshow(gaussian, 'gray')

    v = np.median(img)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv.Canny(gaussian, lower, upper, apertureSize=3)
    plt.subplot(2, 3, 5)
    plt.title("Canny")
    plt.imshow(edges)

    contours, hierarchy = cv.findContours(
        edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 255, 0), -1)
    plt.subplot(2, 3, 6)
    plt.title("contours")
    plt.imshow(img)
    plt.show()

# return y, y+h, x, x+w


def argumentExist():
    try:
        in_img_dir = sys.argv[1]
        out_features = sys.argv[2]
        return in_img_dir, out_features
    except IndexError:
        print(
            "Por favor proporcione ambos archivos, ingresando el grafo primero luego la entrada del problema"
        )
        sys.exit(1)


def main():
    in_img_dir, out_features = argumentExist()

    for img in os.listdir(in_img_dir):
        calculateFeatures(os.path.join(in_img_dir, img))


if __name__ == '__main__':
    main()

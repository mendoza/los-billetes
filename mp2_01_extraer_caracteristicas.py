import cv2 as cv
import numpy as np
import pandas as pd
import os
import sys


def calculateFeatures(path):
    img = cv.imread(path)

    # Making Sure its horizontal
    (h, w) = img.shape[:2]
    while h > w:
        img = np.rot90(img)
        (h, w) = img.shape[:2]

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    guassianBlur = cv.GaussianBlur(gray, (5, 5), 0)

    GaussianCanny = cv.Canny(guassianBlur, 100, 200)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(GaussianCanny, cv.COLOR_GRAY2RGB)
    cdstP = np.copy(cdst)

    # Probabilistic Line Transform
    linesP = cv.HoughLinesP(GaussianCanny, 1, np.pi / 180, 50, 5, 50, 10)

    # Draw the lines

    y1 = 0
    x1 = 0
    y2 = len(img)
    x2 = len(img[0])
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]),
                    (0, 0, 255), 3, cv.LINE_AA)
        cdstP = cv.cvtColor(cdstP, cv.COLOR_RGB2GRAY)
        pts = np.argwhere(cdstP > 0)
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)

    cropped = img[y1:y2, x1:x2]
    cropped = cv.resize(cropped, (320, 192))
    # # Calculate histogram without mask
    hist1 = cv.calcHist([cropped], [0], None, [256], [0, 256])
    hist2 = cv.calcHist([cropped], [1], None, [256], [0, 256])
    hist3 = cv.calcHist([cropped], [2], None, [256], [0, 256])

    r = np.argmax(hist1)
    g = np.argmax(hist2)
    b = np.argmax(hist3)
    return r, g, b, x1, y1, x2, y2


def argumentExist():
    try:
        in_img_dir = sys.argv[1]
        out_features = sys.argv[2]
        return in_img_dir, out_features
    except IndexError:
        print(
            "Por favor proporcione ambos archivos, ingresando el directorio primero luego el nombre del archivo .csv"
        )
        sys.exit(1)


def main():
    in_img_dir, out_features = argumentExist()
    DF = pd.DataFrame({"img": [], "r": [], "g": [], "b": [], "x1": [], "y1": [], "x2": [], "y2": []},
                      columns=['img', 'r', 'g', 'b', 'x1', 'y1', 'x2', 'y2'])

    for img in os.listdir(in_img_dir):
        path = os.path.join(in_img_dir, img)
        r, g, b, x1, y1, x2, y2 = calculateFeatures(path)

        DF = DF.append(
            {"img": path, "r": r, "g": g, "b": b, "x1": x1, "y1": y1, "x2": x2, "y2": y2}, ignore_index=True)

    DF.to_csv(out_features if out_features.endswith(
        ".csv") else out_features+".csv", index=False)


if __name__ == '__main__':
    main()

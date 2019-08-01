import cv2
import glob
import numpy as np

val = 100


def subtract(image, val):
    M = np.ones(image.shape, dtype="uint8") * val
    subtracted = cv2.subtract(image, M)
    return subtracted


def adjust_gamma(image, gamma=0.4):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def read_images(image_path):
    for image in glob.glob(image_path):
        name = image.split('/')[-1]
        original = cv2.imread(image)
        adjusted = adjust_gamma(original)
        subtracted = subtract(original, val)
        # cv2.imwrite('dark/' + name, subtracted)
        cv2.imshow("images", np.hstack([original, adjusted]))
        cv2.waitKey(0)


def main():
    image_path = 'images/*.jpg'
    read_images(image_path)


if __name__ == "__main__":
    main()
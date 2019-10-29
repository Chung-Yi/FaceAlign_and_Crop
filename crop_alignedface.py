import cv2
import os
import gc
import math
import glob
import traceback
import numpy as np
import face_recognition as fr
from os import walk
from keras.models import load_model
from multiprocessing import Pool
from argparse import ArgumentParser
from utils import *

path = os.path.abspath(os.path.dirname(__file__))
model_name = os.path.join(path, 'models/cnn_0702.h5')

parser = ArgumentParser()
parser.add_argument('--input_path', help='choose a input data path')
parser.add_argument('--output_path', help='choose a output data path')
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

INPUT_SIZE = 200
RESIZE = 224
DELTA = 20
types = ('*.jpg', '*.JPG')
landmark_model = load_model(model_name)


def handle_error(e):
    traceback.print_exception(type(e), e, e.__traceback__)


def process_image(image, output_path=output_path):
    gc.collect()
    folder_name = image.split('/')[-2]
    image_name = image.split('/')[-1]

    if (os.path.isdir(os.path.join(output_path, folder_name))) == False:
        os.mkdir(os.path.join(output_path, folder_name))

    face_image = cv2.imread(image)
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    locs = []
    locations = fr.face_locations(face_image_rgb)
    locs = get_max_locations(locations)

    if locs is not None:
        face_img, delta_locs, _ = cut_face(face_image_rgb, locs)

        if face_img.size != 0:
            face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_reshape = np.reshape(face_resized,
                                      (1, INPUT_SIZE, INPUT_SIZE, 3))
            face_normalize = face_reshape.astype('float32') / 255

            points = landmark_model.predict(face_normalize)

            points = np.reshape(points, (-1, 2))

            new_face, new_delta_locs, new_loc = cut_face(
                face_image_rgb, locs, delta=DELTA)

            points[:, 0] *= face_img.shape[1]
            points[:, 1] *= face_img.shape[0]

            points[:, 0] += new_delta_locs[0] - delta_locs[0]
            points[:, 1] += new_delta_locs[1] - delta_locs[1]

            # points[:, 0] -= locs[0]
            # points[:, 1] -= locs[1]

            # points[:, 0] += delta_locs_2[0]
            # points[:, 1] += delta_locs_2[1]
            # points[:, 0] += locs[0] - delta_locs_1[0]
            # points[:, 1] += locs[1] - delta_locs_1[1]

            for point in points:
                cv2.circle(new_face, (int(point[0]), int(point[1])), 1,
                           (0, 255, 0), -1, cv2.LINE_AA)
            cv2.imwrite('output/landmark.jpg', new_face)

            # get aligned face
            faceAligned = align(new_face, points, new_loc)

            # get aligned face locations
            # locations = fr.face_locations(faceAligned)
            # locs = get_max_locations(locations)

            # print(os.path.join(output_path, folder_name, image_name))
            # print(faceAligned.shape)
            # print(locs)
            # if locs is not None:

            # face = cv2.cvtColor(
            #     faceAligned[locs[1]:locs[3], locs[0]s:locs[2]], cv2.COLOR_RGB2BGR)
            # face = cv2.cvtColor(
            #     faceAligned[new_loc[1]:new_loc[3], new_loc[0]:new_loc[2]],
            #     cv2.COLOR_RGB2BGR)

            # # get aligned face points
            # faceAligned_resized = cv2.resize(faceAligned,
            #                                  (INPUT_SIZE, INPUT_SIZE))
            # faceAligned_reshape = np.reshape(faceAligned_resized,
            #                                  (1, INPUT_SIZE, INPUT_SIZE, 3))
            # faceAligned_normalize = faceAligned_reshape.astype('float32') / 255

            # points = landmark_model.predict(faceAligned_normalize)
            # points = np.reshape(points, (-1, 2))

            # points[:, 0] *= faceAligned.shape[1]
            # points[:, 1] *= faceAligned.shape[0]

            # eye_occ, eye_img = eyes_images(faceAligned, points)
            # eye_l_occ = eye_occ[0]
            # eye_r_occ = eye_occ[1]
            # eye_l_img = eye_img[0]
            # eye_r_img = eye_img[1]

            # if (eye_l_occ > 0.18 and eye_r_occ > 0.18) and abs(
            #         np.subtract(eye_l_occ, eye_r_occ)) < 0.3:

            # min_x, min_y, max_x, max_y = get_minimal_box(points)
            # face = crop_landmark_face(points, faceAligned)
            # face = face[min_y - 20:max_y + 20, min_x - 20:max_x + 20]
            # face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            face = cv2.resize(faceAligned[:, :, ::-1], (RESIZE, RESIZE))

            cv2.imwrite(
                os.path.join(output_path, folder_name, image_name), face)


def eyes_images(face, points):

    eye_r_x1 = int(points[42][0])
    eye_r_x2 = int(points[45][0])
    eye_r_y1 = int(
        min(points[42][1], points[43][1], points[44][1], points[45][1]))
    eye_r_y2 = int(
        max(points[42][1], points[47][1], points[46][1], points[45][1]))

    eye_l_x1 = int(points[36][0])
    eye_l_x2 = int(points[39][0])
    eye_l_y1 = int(
        min(points[36][1], points[37][1], points[38][1], points[39][1]))
    eye_l_y2 = int(
        max(points[39][1], points[41][1], points[40][1], points[36][1]))

    eye_l_img = face[eye_l_y1:eye_l_y2, eye_l_x1:eye_l_x2]
    eye_r_img = face[eye_r_y1:eye_r_y2, eye_r_x1:eye_r_x2]

    if (len(eye_l_img) == 0) or (len(eye_r_img) == 0):
        return None, None

    eye_l_area = eye_l_img.shape[0] * eye_l_img.shape[1]
    eye_l_gray = cv2.cvtColor(eye_l_img, cv2.COLOR_RGB2GRAY)
    eye_l_occ = np.count_nonzero(eye_l_gray < 50) / eye_l_area

    eye_r_area = eye_r_img.shape[0] * eye_r_img.shape[1]
    eye_r_gray = cv2.cvtColor(eye_r_img, cv2.COLOR_BGR2GRAY)
    eye_r_occ = np.count_nonzero(eye_r_gray < 50) / eye_r_area

    return (eye_l_occ, eye_r_occ), (eye_l_img, eye_r_img)


def align(image, points, locs):

    desiredFaceWidth = image.shape[1]
    desiredFaceHeight = image.shape[0]
    leftEyePts = points[36:40]
    rightEyePts = points[42:46]

    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))
    scale = 1
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    center = ((locs[0] + locs[2]) / 2, (locs[1] + locs[3]) / 2)
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return output


def main():
    folders = []
    images = []
    for root, dirs, files in walk(input_path):
        folders.append(root)

    for files in types:
        if all(folders):
            images.extend(glob.glob(input_path + '/**/' + files))
        else:
            images.extend(glob.glob(input_path + '/' + files))

    for i, image in enumerate(images):
        process_image(image)
        if (i + 1) % 100 == 0:
            print('{} images are finished'.format(i + 1))

    print('All Done')


if __name__ == "__main__":
    main()
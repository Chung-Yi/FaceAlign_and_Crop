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

path = os.path.abspath(os.path.dirname(__file__))
model_name = os.path.join(path, 'models/cnn_0702.h5')

parser = ArgumentParser()
parser.add_argument('--input_path', help='choose a input data path')
parser.add_argument('--output_path', help='choose a output data path')
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

INPUT_SIZE = 200
types = ('*.jpg', '*.JPG')
landmark_model = load_model(model_name)


def handle_error(e):
    traceback.print_exception(type(e), e, e.__traceback__)


def get_max_locations(locations, locs=None):
    area1 = 0.0
    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        area2 = (end_x - start_x) * (end_y - start_y)
        if area2 > area1:
            area1 = area2
            locs = start_x, start_y, end_x, end_y
    return locs


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
        face_img, delta_locs = cut_face(face_image_rgb, locs)

        if face_img.size != 0:
            face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_reshape = np.reshape(face_resized,
                                      (1, INPUT_SIZE, INPUT_SIZE, 3))
            face_normalize = face_reshape.astype('float32') / 255

            points = landmark_model.predict(face_normalize)

            points = np.reshape(points, (-1, 2))

            points[:, 0] *= face_img.shape[1]
            points[:, 1] *= face_img.shape[0]
            points[:, 0] += locs[0] - delta_locs[0]
            points[:, 1] += locs[1] - delta_locs[1]

            # get aligned face
            faceAligned = align(face_image_rgb, points, locs)

            # get aligned face locations
            # locations = fr.face_locations(faceAligned)
            # locs = get_max_locations(locations)

            # print(os.path.join(output_path, folder_name, image_name))
            # print(faceAligned.shape)
            # print(locs)
            # if locs is not None:
            face = cv2.cvtColor(faceAligned[locs[1]:locs[3], locs[0]:locs[2]],
                                cv2.COLOR_RGB2BGR)

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
            face = cv2.resize(face, (112, 112))

            cv2.imwrite(
                os.path.join(output_path, folder_name, image_name), face)


def draw_landmak_point(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1,
                   cv2.LINE_AA)


def get_minimal_box(points):
    min_x = int(min([point[0] for point in points]))
    max_x = int(max([point[0] for point in points]))
    min_y = int(min([point[1] for point in points]))
    max_y = int(max([point[1] for point in points]))
    return [min_x, min_y, max_x, max_y]


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


def cut_face(face_image, locations):
    face_imgs = []
    delta_locs = []
    width = face_image.shape[1]
    height = face_image.shape[0]

    start_x, start_y, end_x, end_y = locations
    delta = 40

    new_start_y = start_y - delta
    new_end_y = end_y + delta
    new_start_x = start_x - delta
    new_end_x = end_x + delta

    new_start_x, new_start_y, new_end_x, new_end_y = max(0, new_start_x), max(
        0, new_start_y), min(width, new_end_x), min(height, new_end_y)

    face_img = face_image[new_start_y:new_end_y, new_start_x:new_end_x, :]
    delta_locs = (abs(new_start_x - start_x), abs(new_start_y - start_y))
    return face_img, delta_locs


def face_remap(shape):
    remapped_image = shape.copy()
    remapped_image[17] = shape[78]
    remapped_image[18] = shape[74]
    remapped_image[19] = shape[79]
    remapped_image[20] = shape[73]
    remapped_image[21] = shape[72]
    remapped_image[22] = shape[80]
    remapped_image[23] = shape[71]
    remapped_image[24] = shape[70]
    remapped_image[25] = shape[69]
    remapped_image[26] = shape[68]
    remapped_image[27] = shape[76]
    remapped_image[28] = shape[75]
    remapped_image[29] = shape[77]
    remapped_image[30] = shape[0]

    return remapped_image


def crop_landmark_face(points, face_image):
    #initialize mask array and draw mask image
    points_int = np.array([[int(p[0]), int(p[1])] for p in points])
    remapped_shape = np.zeros_like(points)
    landmark_face = np.zeros_like(face_image)
    feature_mask = np.zeros((face_image.shape[0], face_image.shape[1]))

    remapped_shape = face_remap(points_int)
    remapped_shape = cv2.convexHull(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[0:31], 1)
    feature_mask = feature_mask.astype(np.bool)
    landmark_face[feature_mask] = face_image[feature_mask]
    return landmark_face


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
    batch = 1
    for i, image in enumerate(images):
        process_image(image)
        if (i + 1) % 100 == 0:
            print('{} images are finished'.format(i + 1))

    print('All Done')


if __name__ == "__main__":
    main()
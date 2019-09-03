import cv2
import os
import glob
import collections
import face_recognition as fr
from utils import *
from PIL import Image
from keras.models import load_model
from save_csv import write_to_csv

INPUT_SIZE = 200
path = os.path.abspath(os.path.dirname(__file__))
model_name = os.path.join(path, 'models/cnn_0702.h5')
landmark_model = load_model(model_name)


def face_remap_points(shape):

    remapped_image = cv2.convexHull(shape)

    return remapped_image


def crop_left_face(face_image, points):
    points = np.concatenate(
        (points[:9], points[27:31], points[68:72], points[75:78]), axis=0)

    points_int = np.array([[int(p[0]), int(p[1])] for p in points])
    remapped_shape = np.zeros_like(points)
    landmark_face = np.zeros_like(face_image)
    feature_mask = np.zeros((face_image.shape[0], face_image.shape[1]))

    remapped_shape = face_remap_points(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[:], 1)
    feature_mask = feature_mask.astype(np.bool)
    landmark_face[feature_mask] = face_image[feature_mask]
    return landmark_face


def crop_right_face(face_image, points):
    points = np.concatenate(
        (points[8:17], points[71:75], points[78:81], points[27:31]), axis=0)

    points_int = np.array([[int(p[0]), int(p[1])] for p in points])
    remapped_shape = np.zeros_like(points)
    landmark_face = np.zeros_like(face_image)
    feature_mask = np.zeros((face_image.shape[0], face_image.shape[1]))

    remapped_shape = face_remap_points(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[:], 1)
    feature_mask = feature_mask.astype(np.bool)
    landmark_face[feature_mask] = face_image[feature_mask]
    return landmark_face


def crop_landmark(face_image, points):
    #initialize mask array and draw mask image
    points_int = np.array([[int(p[0]), int(p[1])] for p in points])
    remapped_shape = np.zeros_like(points)
    landmark_face = np.zeros_like(face_image)
    feature_mask = np.zeros((face_image.shape[0], face_image.shape[1]))

    remapped_shape = face_remap_points(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[:], 1)
    feature_mask = feature_mask.astype(np.bool)
    landmark_face[feature_mask] = face_image[feature_mask]
    return landmark_face


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


def process_image(image):

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
            # points[:, 0] += locs[0] - delta_locs[0]
            # points[:, 1] += locs[1] - delta_locs[1]
            # points_int = np.array([[int(p[0]), int(p[1])] for p in points])
            # remapped_shape = face_remap_points(points_int)

            landmark_face = crop_landmark(face_img, points)
            left_landmark_face = crop_left_face(face_img, points)
            right_landmark_face = crop_right_face(face_img, points)

            face = cv2.cvtColor(landmark_face, cv2.COLOR_RGB2BGR)
            left_face = cv2.cvtColor(left_landmark_face, cv2.COLOR_RGB2BGR)
            right_face = cv2.cvtColor(right_landmark_face, cv2.COLOR_RGB2BGR)

            # cv2.imshow('img', face)
            # cv2.waitKey(0)
            # cv2.imshow('left_face', left_face)
            # cv2.waitKey(0)
            # cv2.imshow('right_face', right_face)
            # cv2.waitKey(0)

            return left_landmark_face, right_landmark_face, face

    else:

        return None, None, None


def get_pixels(image):
    height = image.shape[0]
    width = image.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            pixel = image[r][c]
            if np.any(pixel != [0, 0, 0]):
                yield pixel


def get_image_brightness(image):
    height = image.shape[0]
    width = image.shape[1]
    total_brightness = 0
    total_pixels = 0
    for r in range(0, height):
        for c in range(0, width):
            pixel = image[r][c]
            if np.any(pixel != [0, 0, 0]):
                red, green, blue = pixel
                redness = red * 0.2126
                greenness = green * 0.7152
                blueness = blue * 0.0722
                brightness = redness + greenness + blueness
                total_brightness += brightness
                total_pixels += 1

    if total_pixels > 0:
        average_brightness = total_brightness / total_pixels
        return round(average_brightness, 3)
    else:
        return 0

    


def get_brightness(image):
    total_pixels = 0
    total_brightness = 0
    for pixel in get_pixels(image):

        brightness = cal_brightness(pixel)
        total_brightness += brightness
        total_pixels += 1

    if total_pixels > 0:
        average_brightness = total_brightness / total_pixels
        return round(average_brightness, 3)
    else:
        return 0


def cal_brightness(pixel):
    red, green, blue = pixel
    redness = red * 0.2126
    greenness = green * 0.7152
    blueness = blue * 0.0722

    brightness = redness + greenness + blueness

    return brightness


def main():
    image_path = 'images/**/*.jpg'
    img_info = []

    for i, image in enumerate(glob.glob(image_path)):

        name = image.split('/')[-1]
        anno = image.split('/')[-2]

        left_face, right_face, face = process_image(image)

        if (left_face is None) and (right_face is None):
            continue
        # l_brightness = get_brightness(left_face)
        # r_brightness = get_brightness(right_face)

        l_brightness = get_image_brightness(left_face)
        r_brightness = get_image_brightness(right_face)
        delta_brightness = abs(l_brightness - r_brightness)
        average_brightness = (l_brightness + r_brightness) / 2

        img_info.append([
            name, l_brightness, r_brightness, anno, delta_brightness,
            average_brightness
        ])
        # cv2.imwrite('crop_face/' + name, face)

        if l_brightness > 100 and r_brightness > 100:
            os.rename(image, os.path.join('images', 'light', name))
        elif l_brightness < 100 and r_brightness < 100:
            os.rename(image, os.path.join('images', 'dark', name))
        elif l_brightness > r_brightness:
            os.rename(image, os.path.join('images', 'right_dark', name))
        elif l_brightness < r_brightness:
            os.rename(image, os.path.join('images', 'left_dark', name))
        else:
            os.rename(image, os.path.join('images', 'none', name))

    print("Start writing in csv file ------------------")

    # write_to_csv(img_info)

    print('Done')


if __name__ == '__main__':
    main()
import numpy as np
import cv2


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


def crop_landmark_face(face_image, points):
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


def get_max_locations(locations, locs=None):
    area1 = 0.0
    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        area2 = (end_x - start_x) * (end_y - start_y)
        if area2 > area1:
            area1 = area2
            locs = start_x, start_y, end_x, end_y
    return locs


def cut_face(face_image, locations, delta=40):
    width = face_image.shape[1]
    height = face_image.shape[0]

    start_x, start_y, end_x, end_y = locations

    new_start_y = start_y - delta
    new_end_y = end_y + delta
    new_start_x = start_x - delta
    new_end_x = end_x + delta

    new_start_x, new_start_y, new_end_x, new_end_y = max(0, new_start_x), max(
        0, new_start_y), min(width, new_end_x), min(height, new_end_y)

    face_img = face_image[new_start_y:new_end_y, new_start_x:new_end_x, :]
    delta_locs = (abs(new_start_x - start_x), abs(new_start_y - start_y))
    return face_img, delta_locs, (new_start_x, new_start_y, new_end_x,
                                  new_end_y)

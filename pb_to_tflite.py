import tensorflow as tf


def main():
    pb_path = 'models/face_landmarks_pb'
    input_node = 'conv2d_1_input'
    output_node = 'dense_3/BiasAdd'

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_path, [input_node], [output_node])

    tflite_model = converter.convert()

    with open('models/converted_model.tflite', "wb") as f:
        f.write(tflite_model)

    # exporting keras h5 file
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(
    #     "models/cnn_0702.h5")
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    main()
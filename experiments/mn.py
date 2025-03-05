# experiments based on
# https://www.tensorflow.org/hub/tutorials/movenet
# https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main
# Lightning for low latency, Thunder for high accuracy


import tensorflow as tf
import tensorflow_hub as hub # install without dependncies --no-deps
import numpy as np
import cv2
import mn_viz2
import time

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')

image_path = 'input_image.jpeg'

def movenet(interpreter, input_image):
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    # print(input_details)
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def movenet1(model, image):
    input_size = 256

    # image: np.ndarray = cv2.imread(image_path)
    input_image: np.ndarray = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.int32)  # int32へキャスト

    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)
    return keypoints_with_scores

def load_ML1():
    module = hub.load("C:/Developer/DepthAI/DepthPose/models/movenet-tensorflow2-multipose-lightning-v1")
    model = module.signatures['serving_default']
    input_size = 256
    return model, input_size

def run_Multi(model, image, input_size):
    image_width, image_height = image.shape[1], image.shape[0]

    input_image: np.ndarray = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.int32)  # int32へキャスト

    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    keypoints_list, scores_list = [], []
    bbox_list = []
    for keypoints_with_score in keypoints_with_scores:
        keypoints = []
        scores = []
        # キーポイント
        for index in range(17):
            keypoint_x = int(image_width *
                             keypoints_with_score[(index * 3) + 1])
            keypoint_y = int(image_height *
                             keypoints_with_score[(index * 3) + 0])
            score = keypoints_with_score[(index * 3) + 2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        bbox_ymin = int(image_height * keypoints_with_score[51])
        bbox_xmin = int(image_width * keypoints_with_score[52])
        bbox_ymax = int(image_height * keypoints_with_score[53])
        bbox_xmax = int(image_width * keypoints_with_score[54])
        bbox_score = keypoints_with_score[55]

        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append(
            [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list

def load_SL4():
    module = hub.load("C:/Developer/DepthAI/DepthPose/models/movenet-tensorflow2-singlepose-lightning-v4")
    model = module.signatures['serving_default']
    input_size = 192
    return model, input_size

def load_ST4():
    module = hub.load("C:/Developer/DepthAI/DepthPose/models/movenet-tensorflow2-singlepose-thunder-v4")
    model = module.signatures['serving_default']
    input_size = 256
    return model, input_size

def run_Single(model, image, input_size):
    image_width, image_height = image.shape[1], image.shape[0]

    input_image: np.ndarray = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.int32)  # int32へキャスト

    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)
        # print(scores)

    return keypoints, scores, None


with tf.device('/device:GPU:0'):
    model, input_size = load_ST4()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # keypoints_with_scores = movenet2(model, frame)

        start_time = time.time()
        # output_overlay  = mn_viz.draw_prediction_on_image(frame, keypoints_with_scores)
        for i in range(6):
            keypoints_list, scores_list, bbox_list = run_Single(model, frame, input_size)
        # print(keypoints_list)
        elapsed_time = time.time() - start_time
        debug_image = mn_viz2.draw_Single(
            frame,
            elapsed_time,
            0.5,
            keypoints_list,
            scores_list,
            0.5,
            bbox_list)

        cv2.imshow('MoveNet(multipose) Demo', debug_image )
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

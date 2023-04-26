from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from data_proc import ExtractFeatures
from models import Inference
from utils.draw_utils import DrawUtils
import time

mp_solution = mp.solutions
model_path = "./saved_models/trained_classifier_allclass_wins40.pickle"
action_labels = [
    "crafty_tricks",
    "sowing_corn_and_driving_pigeons",
    "waves_crashing",
    "flower_clock",
    "wind_that_shakes_trees",
    "big_wind",
    "bokbulbok",
    "seaweed_in_the_swell_sea",
    "chulong_chulong_phaldo",
    "chalseok_chalseok_phaldo",
]

infer = Inference(model_path, action_labels)
extract = ExtractFeatures()
drawutils = DrawUtils()


def prediction(path):
    cap = cv2.VideoCapture(path)

    with mp_solution.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        inference_pose_deque = deque()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter(
            "predicted_video_02.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, size
        )
        t1 = 0
        t2 = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_solution.drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_solution.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_solution.drawing_styles.get_default_pose_landmarks_style(),
            )

            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark

            # write data to the json file
            frame_landmarks = {
                str(index): [keypoint.x, keypoint.y]
                for index, keypoint in enumerate(landmarks)
            }
            scaled_skeleton = extract.scaler(frame_landmarks)
            frame_landmarks_body_only_scaled = extract.get_body_points_inference(
                scaled_skeleton
            )

            inference_pose_deque.append(frame_landmarks_body_only_scaled[0])
            extract.maintain_deque_size(inference_pose_deque)

            if len(inference_pose_deque) == 5:
                predicted_label, predicted_score = infer.predict(inference_pose_deque)
                t2 = time.time()
                fps = 1.0 / (t2 - t1)
                t1 = t2
                minx, miny, maxx, maxy = drawutils.draw_boundingbox(
                    image, frame_landmarks
                )
                image = cv2.rectangle(image, (minx, miny), (maxx, maxy), (0, 255, 0), 4)
                cv2.putText(
                    image,
                    str("FPS: {:.0f}".format(fps)),
                    (1, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0, 0, 0),
                    1,
                )
                cv2.putText(
                    image,
                    f"{predicted_label}",
                    (1, 100),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.75,
                    (255, 0, 0),
                    1,
                )
                result.write(image)
                cv2.imshow("Action recognition system", image)
                if cv2.waitKey(5) & 0xFF == "q":
                    break
    cap.release()


# path = "/media/lakpa/Storage/youngdusan_data/youngdusan_video_data/bokbulbok/bokbulbok_5.mov"
# path = "/media/lakpa/Storage/youngdusan_data/youngdusan_video_data/big_wind/big_wind_10.mov"
path = "/media/lakpa/Storage/youngdusan_data/test_video/videoplayback.mp4"
prediction(path)

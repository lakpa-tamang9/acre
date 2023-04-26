import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from data_proc import ExtractFeatures
from models import Inference
from utils.draw_utils import DrawUtils

mp_solution = mp.solutions
parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type= str, help = "Path to the trained model (.pickle)")
parser.add_argument("--labels_path", type=str, help="Path to the labels file (.txt)")
parser.add_argument("--video_path", type=str, help="Path to your video file for inference.")
parser.add_argument("--write_output", default=False, type = bool, help = "Set to true if you want to save the predicted actions in a video file.")
parser.add_argument("--output_path", default =None, type=str, help="Path to your output directory, where you want to save the predicted video.")
args = parser.parse_args()

with open("./labels.txt") as f:
    action_labels = [label.strip() for label in f.read().split("\n")]

infer = Inference(args.model_path, action_labels)
extract = ExtractFeatures()
drawutils = DrawUtils()


def prediction(path):
    cap = cv2.VideoCapture(0 if path == "0" else path)

    with mp_solution.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        inference_pose_deque = deque()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter(
            "{}/acre_output.avi".format(args.output_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size
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
                if args.write_output:
                    result.write(image)
                cv2.imshow("Action recognition system", image)
                if cv2.waitKey(5) & 0xFF == "q":
                    break
    cap.release()

if __name__=="__main__":
    prediction(args.video_path)

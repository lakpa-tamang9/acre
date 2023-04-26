from itertools import count
import mediapipe as mp
import cv2
import json
import os
import natsort
from tqdm import tqdm
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def run_mediapipe_pose(path):
    for root, _, files in os.walk(path):
        for count, file in tqdm(natsort.natsorted(enumerate(files))):
            pose_data = {}
            if file.endswith((".mov", ".mp4")):
                class_name = root.split("/")[-1]
                filename_list = file.split(".")[0].split("_")[:-1]
                filename_list.append(str(count))
                filename = ("_").join(filename_list)

                cap = cv2.VideoCapture(os.path.join(root, file))

                output_path = (
                    ("/").join(path.split("/")[:-1])
                    + "/"
                    + "landmarks_data"
                    + "/"
                    + f"{class_name}"
                )
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                video_landmarks = []
                frame_landmarks_dict = {}

                with mp_pose.Pose(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                ) as pose:
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            # If loading a video, use 'break' instead of 'continue'.
                            break

                        # Flip the image horizontally for a later selfie-view display, and convert
                        # the BGR image to RGB.
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        image.flags.writeable = False
                        results = pose.process(image)
                        if not results.pose_landmarks:
                            continue

                        landmarks = results.pose_landmarks.landmark

                        # write data to the json file
                        frame_landmarks = {
                            str(index): [keypoint.x, keypoint.y]
                            for index, keypoint in enumerate(landmarks)
                        }

                        # frame_landmarks_dict[count] = frame_landmarks
                        video_landmarks.append(frame_landmarks)
                pose_data[filename] = video_landmarks
            cap.release()

            with open(f"{output_path}/{filename}.json", "w") as f:
                json.dump(pose_data, f)


run_mediapipe_pose(path="/media/lakpa/Storage/youngdusan_data/youngdusan_video_data")

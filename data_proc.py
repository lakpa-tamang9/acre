import argparse
from ftplib import all_errors
import json
import os
import string
import time
from collections import deque
from typing import Tuple
from tqdm import tqdm
import natsort
import numpy as np


class DataPreProc:
    def __init__(self):
        self.NOSE = 0
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.KEYPOINT_INDICES_TO_REMOVE = (
            [i for i in range(1, 11)] + [i for i in range(17, 23)] + [29, 30, 31, 32]
        )

    def get_example_skeleton(self, path: string) -> list:
        """Converts the pose data in json format of all video files and appends them to list.

        Returns:
            list: Pose data of all video files of all classes
        """
        all_files_info = []
        start = time.time()
        files_count = {}
        for root, _, files in os.walk(path):
            files_count[root.split("/")[-1]] = len(files)
            for file in natsort.natsorted(files):

                if file.endswith(".json"):
                    filename = file.split(".")[0]
                    with open(os.path.join(root, file), "r") as f:
                        example_skeleton = json.load(f)
                        if len(example_skeleton) == 0:
                            print(f"Ignoring {filename} because it is empty!!!")
                            continue
                    list(example_skeleton.values())[0].insert(0, filename)
                    all_files_info.append(example_skeleton)
        end = time.time()

        print(
            f"Data load completed !!! \n Took {end - start} seconds to process the files!"
        )
        return all_files_info

    def fill_missing_vals(self):
        video_pose_values = self.get_example_skeleton(path="./data")

        for frame_pose_values in video_pose_values:
            actual_indices = [i for i in range(33)]
            for detected_index in set(frame_pose_values.keys()):
                if detected_index in actual_indices:
                    return self.get_scaled_skeleton(video_pose_values=video_pose_values)
            else:
                pass
            return None

    def skip_n_frames(
        self, all_class_frames: list, required_video_frames=40, skip_frames=5
    ) -> list:
        """Skips certain frames in a video file. Then makes the total number of
        frames uniform by removing extra frames or copying the existing
        frames accordingly.

        Args:
            all_class_frames (list): Pose data of all files of all classes
            required_video_frames (int) : How many frames to fix in a single video. Defaults to 60.
            skip_frames (int): Skip value for indexing the video frames
                            (For eg: [x, (x + skip_frames) ...]). Defaults to 5.

        Returns:
            list: Pose data of all video of all class with certain frames skipped
        """
        skipped_video_frames_all_class = []
        equivalent_video_frames = []
        for one_class_frames in all_class_frames:
            skipped_video_frames = []
            frames = list(one_class_frames.values())
            for i in range(0, len(frames[0]), skip_frames):
                skipped_video_frames.append(frames[0][i])
            skipped_video_frames_all_class.append(skipped_video_frames)

        for skipped_video_frames in skipped_video_frames_all_class:
            # skipped_video_frames = deque(skipped_video_frames)
            total_video_frames = len(skipped_video_frames)

            # Chek how many frames are extra
            extra_frames = required_video_frames - total_video_frames

            if extra_frames < 0:
                # Remove extra frames according to the index by skipping 5 frames
                remove_index = [i for i in range(1, int(abs(extra_frames) / 2) + 1)]
                for i in remove_index:
                    del skipped_video_frames[i]
                    del skipped_video_frames[-i]
                    # skipped_video_frames.pop()
                    # skipped_video_frames.popleft()
                equivalent_video_frames.append(skipped_video_frames)
            else:
                # Copy the existing frames in the certain indices by skipping 5 frames
                copy_index = [i for i in range(1, abs(extra_frames) + 1)]
                for i in copy_index:
                    copied_frame = skipped_video_frames[i].copy()
                    skipped_video_frames.append(copied_frame)
                equivalent_video_frames.append(skipped_video_frames)

        return equivalent_video_frames

    def scaler(self, frame_poses):
        y_nose = frame_poses.get(str(self.NOSE))[1]
        y_left_hip = frame_poses.get(str(self.LEFT_HIP))[1]
        y_right_hip = frame_poses.get(str(self.RIGHT_HIP))[1]

        y_central_hip = (y_left_hip + y_right_hip) / 2
        height = abs(y_nose - y_central_hip)
        normalized_pose_values = [
            value / height
            for frame_pose_value in list(frame_poses.values())
            for value in frame_pose_value
        ]
        nested_norm_pose_values = {
            str(int(index / 2)): normalized_pose_values[index : index + 2]
            for index in range(len(normalized_pose_values))
            if index % 2 == 0
        }
        return nested_norm_pose_values

    def get_scaled_skeleton(self, all_video_pose_values) -> list:
        """Uses height between nose and hip values to scale the pose keypoints

        Returns:
            list: Scaled pose data of all videos of all classes
        """
        all_video_scaled_values = []

        for single_video_pose_values in all_video_pose_values:
            single_video_scaled_values = []

            for frame_pose_values in single_video_pose_values[1:]:
                nested_norm_pose_values = self.scaler(frame_pose_values)
                single_video_scaled_values.append(nested_norm_pose_values)

            single_video_scaled_values.insert(0, single_video_pose_values[0])
            all_video_scaled_values.append(single_video_scaled_values)

        return all_video_scaled_values

    def get_body_points_inference(self, scaled_keypoints):
        only_body_pose_dict = {}
        body_points_for_inference = []
        for index, values in enumerate(list(scaled_keypoints.values())):
            if index not in self.KEYPOINT_INDICES_TO_REMOVE:
                only_body_pose_dict[str(index)] = values

        body_points_for_inference.append(only_body_pose_dict)
        return body_points_for_inference

    def get_only_body_points(self, all_video_scaled_data) -> list:
        """Removes the unwanted landmarks from the scaled pose values

        Returns:
            list: Pose data of all videos of all classes without face and hands
        """

        # Remove face landmarks except nose + Hand landmarks

        only_body_all_video_poses = []

        for single_video_scaled_data in all_video_scaled_data:
            only_body_single_video_poses = []

            for normalized_frame_pose_list in single_video_scaled_data[1:]:
                only_body_pose_dict = {}

                for index, values in enumerate(
                    list(normalized_frame_pose_list.values())
                ):
                    if index not in self.KEYPOINT_INDICES_TO_REMOVE:
                        only_body_pose_dict[str(index)] = values

                only_body_single_video_poses.append(only_body_pose_dict)
            only_body_single_video_poses.insert(0, single_video_scaled_data[0])
            only_body_all_video_poses.append(only_body_single_video_poses)

        return only_body_all_video_poses

    def check_shoulder_and_leg(self):
        # TODO: Modify and add this method for real time testing with webcam/video
        # TODO: Check the shoulder and leg values. If not exists then discard the frame.
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        return


class ExtractFeatures(DataPreProc):
    def __init__(self):

        self.window_size = 5
        self.pose_deque = deque()

        super().__init__()

    def maintain_deque_size(self, pose_deque) -> None:
        """Performs FIFO if the length of deque exceeds the window size."""
        if len(pose_deque) > self.window_size:
            pose_deque.popleft()

    def get_body_height(self, frame_pose_data: dict) -> int:
        """Calculates the height in each skeletal pose frame using the nose and hip landmarks.

        Args:
            frame_pose_data (sict): Dictionary of preprocessed pose data in a frame.

        Returns:
            int: Height in each frame
        """
        nose_landmarks = frame_pose_data.get(str(self.NOSE))
        left_hip_landmarks = frame_pose_data.get(str(self.LEFT_HIP))
        right_hip_landmarks = frame_pose_data.get(str(self.RIGHT_HIP))

        central_hip_landmark = [
            (left_hip_landmarks[0] + right_hip_landmarks[0]) / 2,
            (left_hip_landmarks[1] + right_hip_landmarks[1]) / 2,
        ]
        height = np.linalg.norm(
            np.array(nose_landmarks) - np.array(central_hip_landmark)
        )
        return height

    def get_pose_feature(self, deque_data: deque) -> np.array:
        """Calculates the pose features of the current frame.
        Pose feature is defined as the array of landmarks (x, y) values
        present in a frame.

        Args:
            deque_data (deque): Pose deque containing pose data as a dictionary type.

        Returns:
            numpy array: Array of pose features.
        """
        features = []
        for i in range(len(deque_data)):
            next_feature = list(deque_data[i].values())
            features += next_feature
        features = np.array([keypoint for feature in features for keypoint in feature])

        return features

    def get_motion_feature(
        self, pose_deque: list, use_nose_only: bool, step: int
    ) -> np.array:
        """Calculates the motion feature between two consecutive frames.

                - Motion feature is defined as the array of difference between
                the landmarks (x, y)of the current frame with the next frame.
                - If use_nose_bool set to True, calculates only the motion feature
                of nose in consecutive frames of the deque, otherwise calculates motion
                feature of all landmarks.

        Args:
            pose_deque (deque): Pose deque containing pose data as a dictionary type.
            use_nose_only (bool): Set True to use nose landmarks only.
            step (int): How many frame steps to use to realize two frames as consecutive frames.

        Returns:
            numpy array: Array of motion feature.
        """
        motion_feature = []

        for i in range(0, len(pose_deque) - step, step):

            current_frame = list(pose_deque[i].values())
            next_frame = list(pose_deque[i + step].values())

            if use_nose_only:
                # Subtract the x, y landmarks from next frame with current frame
                per_frame_pose_difference = np.array(
                    [val for pose in next_frame for val in pose][0:2]
                ) - np.array([val for pose in current_frame for val in pose][0:2])
            else:
                per_frame_pose_difference = np.array(
                    [val for pose in next_frame for val in pose][:]
                ) - np.array([val for pose in current_frame for val in pose][:])

            motion_feature += per_frame_pose_difference.tolist()

        return np.array(motion_feature)

    def features_for_inference(self, video_frame_pose, repeat_val=10):
        # inference_pose_deque = deque()
        # inference_pose_deque.append(video_frame_pose)
        # self.maintain_deque_size(inference_pose_deque)

        # if len(inference_pose_deque) < self.window_size:
        #     return False, None
        # else:
        height_list = [self.get_body_height(frame) for frame in video_frame_pose]
        mean_height = np.mean(height_list)

        # Get pose features
        pose_feature = self.get_pose_feature(video_frame_pose)

        # Get motion features and repeat n times to add weight
        motion_feature_nose = np.repeat(
            (
                self.get_motion_feature(video_frame_pose, use_nose_only=True, step=1)
                / mean_height
            ),
            repeat_val,
        )
        motion_feature_all = self.get_motion_feature(
            video_frame_pose, use_nose_only=False, step=1
        )

        single_frame_features = np.concatenate(
            (pose_feature, motion_feature_nose, motion_feature_all)
        )
        return True, single_frame_features

    def extract_features(self, save_npy: bool, save_csv: bool, repeat_val=10) -> None:
        """This method performs following tasks:
            - Extracts the pose data with only body points
            - Extracts pose, and motion features
            - Creates a final feature by concatenating all individual features
            - Creates a training data and its label and save it as .csv format

        Args:
            repeat_val (int, optional): Adds weight to the motion feature of the nose
            by repeating the array. Defaults to 10.

        """
        pars = argparse.ArgumentParser()
        pars.add_argument("--data_path", help="Path to the json pose data")
        args = pars.parse_args()
        args.data_path = (
            "/media/lakpa/Storage/youngdusan_data/extracted_landmarks_data_json"
        )

        all_video_pose_values_no_skip = self.get_example_skeleton(path=args.data_path)
        all_video_pose_values = self.skip_n_frames(all_video_pose_values_no_skip)

        all_video_scaled_data = self.get_scaled_skeleton(all_video_pose_values)

        output_path = ("/").join(args.data_path.split("/")[:-1])
        videos_poses = self.get_only_body_points(all_video_scaled_data)
        total_frames = []

        for i in range(len(videos_poses)):
            total_frames.append(len(videos_poses[i]) - 1)

        print("******Starting feature extraction*****")

        all_features_with_name = []
        all_features_without_name = []
        all_labels = []
        for index, video_poses in tqdm(enumerate(videos_poses)):
            class_name = ("_").join(video_poses[0].split("_")[:-1])
            video_count = int(video_poses[0].split("_")[-1]) + 1
            all_labels.extend([video_poses[0]] * (len(videos_poses[index]) - 1))
            for frame_count, frame_poses in enumerate(video_poses[1:]):
                self.pose_deque.append(frame_poses)
                if len(self.pose_deque) < self.window_size:
                    continue
                self.maintain_deque_size(self.pose_deque)

                height_list = [self.get_body_height(frame) for frame in self.pose_deque]
                mean_height = np.mean(height_list)

                # Get pose features
                pose_feature = self.get_pose_feature(self.pose_deque)

                # Get motion features and repeat n times to add weight
                motion_feature_nose = np.repeat(
                    (
                        self.get_motion_feature(
                            self.pose_deque, use_nose_only=True, step=1
                        )
                        / mean_height
                    ),
                    repeat_val,
                )
                motion_feature_all = self.get_motion_feature(
                    self.pose_deque, use_nose_only=False, step=1
                )
                # Combine all features
                # all_features_with_name for csv data
                # all_features_without_name for numpy array data

                single_frame_features = np.concatenate(
                    (pose_feature, motion_feature_nose, motion_feature_all)
                )
                single_frame_features_with_name = single_frame_features.tolist()
                single_frame_features_with_name.insert(0, video_poses[0])
                all_features_with_name.append(np.array(single_frame_features_with_name))

                all_features_without_name.append(single_frame_features)
                # frame_count =
                if save_npy:
                    npy_path = (
                        f"{output_path}/npy_data/{class_name}/{str(video_count)}/"
                    )
                    if not os.path.exists(npy_path):
                        os.makedirs(npy_path)

                    np.save(
                        f"{npy_path}/{str(int(frame_count) + 1)}.npy",
                        single_frame_features,
                    )
        if save_csv:
            # Create label dict with corresponding label value
            label_dict = {}
            labels = []
            for label in all_labels:
                label_name = ("_").join(label.split("_")[:-1])
                labels.append(label_name)

            # Get unique labels from the labels list
            unique_labels = list(dict.fromkeys(labels))

            for index, label in enumerate(unique_labels):
                label_dict[label] = index

            # Create categorical labels
            categorical_label = []
            for label in all_labels:
                label_name = ("_").join(label.split("_")[:-1])
                # for key in list(label_dict.keys()):
                for unique_label in unique_labels:
                    if unique_label == label_name:
                        new_label_name = label_name.replace(
                            label_name, str(label_dict[label_name])
                        )
                        categorical_label.append(int(new_label_name))

            print("Saving features and labels")
            np.savetxt(
                f"{output_path}/features.csv", all_features_without_name, fmt="%.5f"
            )
            np.savetxt(f"{output_path}/labels.csv", categorical_label, fmt="%i")
            print("Save completed!!!")

            print("Feature Extraction Completed!!!")


extract = ExtractFeatures()

extract.extract_features(save_npy=True, save_csv=False)

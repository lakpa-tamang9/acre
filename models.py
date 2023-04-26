import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils.draw_utils import DrawUtils
from data_proc import ExtractFeatures
from collections import deque


class Train(object):
    def __init__(self):
        self._init_all_models()
        self.model = MLPClassifier((20, 30, 40))
        self.draw_utils = DrawUtils()
        self.clf = self._choose_model("Neural Net")

    def predict(self, X):
        """Predict the class index of the feature X"""
        Y_predict = self.clf.predict(self.pca.transform(X))
        return Y_predict

    def predict_and_evaluate(self, te_X, te_Y):
        """Test model on test set and obtain accuracy"""
        te_Y_predict = self.predict(te_X)
        N = len(te_Y)
        n = sum(te_Y_predict == te_Y)
        accu = n / N
        return accu, te_Y_predict

    def train(self, X, Y):
        """Train model. The result is saved into self.clf"""
        NUM_FEATURES_FROM_PCA = 50
        n_components = min(NUM_FEATURES_FROM_PCA, X.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(X)
        # print("Sum eig values:", np.sum(self.pca.singular_values_))
        print("Sum eig values:", np.sum(self.pca.explained_variance_ratio_))
        X_new = self.pca.transform(X)
        print("After PCA, X.shape = ", X_new.shape)
        self.clf.fit(X_new, Y)

    def _choose_model(self, name):
        self.model_name = name
        idx = self.names.index(name)
        return self.classifiers[idx]

    def _init_all_models(self):
        self.names = ["Neural Net"]
        self.model_name = None
        self.classifiers = [MLPClassifier((20, 30, 40))]

    def _predict_proba(self, X):
        """Predict the probability of feature X belonging to each of the class Y[i]"""
        Y_probs = self.clf.predict_proba(self.pca.transform(X))
        return Y_probs  # np.array with a length of len(classes)


class Inference(object):
    def __init__(self, model_path, action_labels):
        self.scores_hist = deque()
        self.DEQUE_MAX_SIZE = 2
        self.data_proc = ExtractFeatures()
        self.action_labels = action_labels
        self.LABEL_UNKNOWN = ""
        self.THRESHOLD_SCORE_FOR_DISP = 0.5

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, pose_data):
        """Predicts the label and score of the input pose data

        Args:
            pose_data (list): Skeletal pose data from mediapipe

        Returns:
            string, array: Predicted label as string, and the scores as array
        """
        good_features, features = self.data_proc.features_for_inference(pose_data)

        if good_features:
            features = features.reshape(-1, features.shape[0])
            curr_scores = curr_scores = self.model._predict_proba(features)[0]
            self.scores = self.score_smoothing(curr_scores)

            if (
                self.scores.max() < self.THRESHOLD_SCORE_FOR_DISP
            ):  # If lower than threshold, bad
                predicted_label = self.LABEL_UNKNOWN
            else:
                predicted_idx = self.scores.argmax()
                predicted_label = self.action_labels[predicted_idx]
        else:
            predicted_label = self.LABEL_UNKNOWN

        return predicted_label, self.scores

    def score_smoothing(self, curr_scores):
        """Smooths the predicted score using sum and multiplication

        Args:
            curr_scores (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.scores_hist.append(curr_scores)
        if len(self.scores_hist) > self.DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

        if 1:  # Use sum
            score_sums = np.zeros((len(self.action_labels),))
            for score in self.scores_hist:
                score_sums += score
            score_sums /= len(self.scores_hist)
            return score_sums

        else:  # Use multiply
            score_mul = np.ones((len(self.action_labels),))
            for score in self.scores_hist:
                score_mul *= score
            return score_mul

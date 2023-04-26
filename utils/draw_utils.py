import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class DrawUtils:
    def __init__(self) -> None:
        pass

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        classes,
        normalize=False,
        title=None,
        cmap=plt.cm.Blues,
        size=None,
    ):
        """(Copied from sklearn website)
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Display normalized confusion matrix ...")
        else:
            print("Display confusion matrix without normalization ...")

        fig, ax = plt.subplots()
        if size is None:
            size = (12, 8)
        fig.set_size_inches(size[0], size[1])

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        ax.set_ylim([-0.5, len(classes) - 0.5])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        return ax, cm

    def draw_boundingbox(self, image: np.array, skeleton: list):
        """Draws bounding box around the image

        Args:
            image (tensor): Input image where bounding box is to be drawn
            skeleton (list): Exctracted pose skeleton from mediapipe

        Returns:
            ints: minimum and maximum values of x and y for bounding boxes
        """
        pose_skeleton = []
        for values in list(skeleton.values()):
            pose_skeleton.append(values)
        pose_skeleton_flattened = [
            pose for skeleton in pose_skeleton for pose in skeleton
        ]
        # Set initial values for min and max of x and y
        minx = 999
        miny = 999
        maxx = -999
        maxy = -999
        i = 0
        NaN = 0

        while i < len(pose_skeleton_flattened):
            if not (
                pose_skeleton_flattened[i] == NaN
                or pose_skeleton_flattened[i + 1] == NaN
            ):
                minx = min(minx, pose_skeleton_flattened[i])
                maxx = max(maxx, pose_skeleton_flattened[i])
                miny = min(miny, pose_skeleton_flattened[i + 1])
                maxy = max(maxy, pose_skeleton_flattened[i + 1])
            i += 2

        # Scale the min and max value according to image shape
        minx = int(minx * image.shape[1])
        miny = int(miny * image.shape[0])
        maxx = int(maxx * image.shape[1])
        maxy = int(maxy * image.shape[0])

        return minx, miny, maxx, maxy

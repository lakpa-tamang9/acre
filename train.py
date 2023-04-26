import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report
from models import Train
import os
from utils.draw_utils import DrawUtils

drawutils = DrawUtils()


def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    """Evaluate accuracy and time cost"""

    # Accuracy
    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(
        classification_report(
            te_Y, te_Y_predict, target_names=classes, output_dict=False
        )
    )

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: " "{:.5f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = drawutils.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8)
    )
    plt.show()


def main():
    """Trains the MLP using numpy array of features and labels

    Args:
        features_path (path): Path to the features csv file
        label_path (path): Path to the label csv file
        pca_features (int): Length of desired PCA dimension
    """
    CLASSES = [
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
    NP_CLASSES = np.array(CLASSES)

    features_path = "/media/lakpa/Storage/youngdusan_data/features.csv"
    label_path = "/media/lakpa/Storage/youngdusan_data/labels.csv"
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(features_path, dtype=float)  # features
    Y = np.loadtxt(label_path, dtype=int)  # labels

    tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.3, random_state=1
    )

    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    print("\nStart training model ...")

    model = Train()
    model.train(tr_X, tr_Y)
    print(model)

    print("\nStart evaluating model ...")

    # evaluate = Evaluate(pca_features)
    evaluate_model(model, NP_CLASSES, tr_X, tr_Y, te_X, te_Y)

    # -- Save model
    print("Saving model...")
    model_output_path = "./saved_models"
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    with open(f"{model_output_path}/trained_classifier_allclass.pickle", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()

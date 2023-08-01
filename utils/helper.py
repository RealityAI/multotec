import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import librosa
from math import log, ceil, floor
import math
import pickle
import re


# import pywt

def plot_roc(y_test, y_score, _names, n_classes=4, prefix="", suffix=""):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(0).clf()
    for i in range(n_classes):
        # plt.figure()
        plt.plot(fpr[i], tpr[i], label=_names[i] + ' (%0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(prefix + ' ROC ' + suffix)
        plt.legend(loc="lower right")
    plt.legend(loc=0)
    plt.show()


def plot_multiclass_roc(y_score, y_test, n_classes, figsize=(17, 6)):
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    # y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    # sns.despine()
    plt.show()


def stratified_sampling(df, strata_col, sample_size):
    _groups = df.groupby(strata_col)
    sample = pd.DataFrame()
    for _, group in _groups:
        stratum_sample = group.sample(frac=sample_size, replace=False, random_state=7)
        sample = pd.concat([sample, stratum_sample], ignore_index=True)
    return sample


def evaluate_model(y_test, y_score):
    y_test = np.argmax(y_test, axis=1)
    y_score = np.argmax(y_score, axis=1)

    cm = confusion_matrix(y_test, y_score, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['air_conditioner', 'car_horn', 'engine_idling', 'siren'])
    disp.plot()
    plt.show()
    # print(confusion_matrix(y_test, y_score))
    # print(classification_report(y_test, y_score))


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def pre_process_signal(y, sr, low=32, high=499):
    y = librosa.effects.preemphasis(y, coef=0.96)

    low_pass = butter_bandpass_filter(y, low, high, sr, order=3)
    # med_pass = butter_bandpass_filter(y, 256, 512, sr, order=3)
    # high_pass = butter_bandpass_filter(y, 1024, 2048, sr, order=3)
    if np.isnan(low_pass).sum() > 0:
        return y
    return low_pass


def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return min(possible_results, key=lambda z: abs(x - 2 ** z))
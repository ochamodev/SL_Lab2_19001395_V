import pandas as pd
from scipy import stats
import numpy as np
from constants import Constants
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score, classification_report

from classification_metrics_model import ClassificationMetricsModel


def remove_outliers_zscore(df: pd.DataFrame, threshold=3):
    z_scores = stats.zscore(df.select_dtypes(Constants.numerics))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    df_filtered = df[filtered_entries]
    return df_filtered


def print_outliers(df: pd.DataFrame, threshold=3):
    z_scores = stats.zscore(df.select_dtypes(Constants.numerics))
    zScores = z_scores.apply(stats.zscore)
    outliers = (zScores.abs() > 3).sum()
    proportion = outliers / (df.shape[0])
    print("Percentage that outliers represent")
    print(proportion * 100)


def calculate_metrics(y_test, y_pred) -> ClassificationMetricsModel:
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    vpn = tn / (tn + fn)
    specificity = tn / (tn + fp)

    return ClassificationMetricsModel(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1 = f1,
        vpn = vpn,
        specificity = specificity
    )

def print_metrics(metrics: ClassificationMetricsModel):
    print(f"Accuracy: {metrics.accuracy}")
    print(f"Precision: {metrics.precision}")
    print(f"Recall: {metrics.recall}")
    print(f"F1: {metrics.f1}")
    print(f"vpn: {metrics.vpn}")
    print(f"Specificity: {metrics.specificity}")

        
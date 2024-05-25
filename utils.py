import pandas as pd
from scipy import stats
import numpy as np
from constants import Constants

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
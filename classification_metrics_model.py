from dataclasses import dataclass
import numpy as np

@dataclass
class ClassificationMetricsModel:
    accuracy: np.float64
    precision: np.float64
    recall: np.float64
    f1: np.float64
    vpn: np.float64
    specificity: np.float64

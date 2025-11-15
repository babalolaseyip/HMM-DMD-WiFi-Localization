"""Localization evaluation metrics."""
import numpy as np
from typing import Dict, List, Tuple


class LocalizationEvaluator:
    def __init__(self, error_thresholds: List[float] = None):
        self.error_thresholds = error_thresholds or [1, 2, 4, 6]

    def mean_localization_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        return float(np.mean(errors))

    def median_localization_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        return float(np.median(errors))

    def percentile_error(self, y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 95) -> float:
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        return float(np.percentile(errors, percentile))

    def accuracy_within_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        return float(np.mean(errors <= threshold))

    def cumulative_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        return sorted_errors, cdf

    def comprehensive_evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        results = {}
        results['mean_error'] = self.mean_localization_error(y_true, y_pred)
        results['median_error'] = self.median_localization_error(y_true, y_pred)
        results['95th_percentile_error'] = self.percentile_error(y_true, y_pred, 95)
        for threshold in self.error_thresholds:
            accuracy = self.accuracy_within_threshold(y_true, y_pred, threshold)
            results[f'accuracy_within_{threshold}m'] = accuracy
        errors, cdf = self.cumulative_distribution(y_true, y_pred)
        results['cdf_errors'] = errors.tolist()
        results['cdf_values'] = cdf.tolist()
        return results

    def print_results(self, results: Dict[str, float]):
        print("\n" + "="*60)
        print("LOCALIZATION EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Error:              {results['mean_error']:.2f} m")
        print(f"Median Error:            {results['median_error']:.2f} m")
        print(f"95th Percentile Error:   {results['95th_percentile_error']:.2f} m")
        print("\nAccuracy within distance thresholds:")
        for threshold in self.error_thresholds:
            key = f'accuracy_within_{threshold}m'
            if key in results:
                print(f"  Within {threshold}m:  {results[key]*100:.2f}%")
        print("="*60 + "\n")

#!/usr/bin/env python3
"""Evaluation script for comparing different localization methods."""
import argparse
import json
import numpy as np
from pathlib import Path
from models.hmm_dmd import HMMDMDLocalizer
from src.data.data_loader import CrawdadDataLoader
from src.data.preprocessor import RSSIPreprocessor
from src.evaluation.metrics import LocalizationEvaluator
from src.evaluation.visualization import plot_cdf_comparison
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate localization models')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Starting model evaluation...")

    data_loader = CrawdadDataLoader()
    X, y, rooms = data_loader.create_synthetic_data(n_samples=500)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    rooms_train = rooms[:split]

    preprocessor = RSSIPreprocessor()
    X_train = preprocessor.preprocess(X_train, fit=True)
    X_test = preprocessor.preprocess(X_test)

    logger.info("Training HMM-DMD...")
    hmm_dmd = HMMDMDLocalizer(dmd_rank=7, n_components=10)
    hmm_dmd.fit(X_train, y_train, rooms_train)
    pred_hmm_dmd = hmm_dmd.predict(X_test)

    evaluator = LocalizationEvaluator()
    results = {'HMM-DMD': evaluator.comprehensive_evaluate(y_test, pred_hmm_dmd)}

    for method, result in results.items():
        print(f"\n{method} Results:")
        evaluator.print_results(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()

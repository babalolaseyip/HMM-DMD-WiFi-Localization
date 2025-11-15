#!/usr/bin/env python3
"""Training script for HMM-DMD localization model."""
import argparse
import json
import numpy as np
from pathlib import Path
from models.hmm_dmd import HMMDMDLocalizer
from src.data.data_loader import CrawdadDataLoader
from src.data.preprocessor import RSSIPreprocessor
from src.evaluation.metrics import LocalizationEvaluator
from src.evaluation.visualization import plot_results
from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Train HMM-DMD WiFi localization model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to configuration file')
    parser.add_argument('--use-synthetic', action='store_true', help='Use synthetic data for testing')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()

    logger = setup_logger(log_file=f'{args.output_dir}/training.log')
    logger.info("Starting HMM-DMD training pipeline...")

    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}")
        logger.info("Using default configuration")
        config = {
            'model': {
                'hmm_dmd': {
                    'dmd_rank': 7,
                    'n_components': 14,
                    'n_mix': 3,
                    'cov_type': 'diag'
                }
            },
            'training': {
                'random_state': 42
            }
        }

    logger.info("Loading data...")
    data_loader = CrawdadDataLoader()

    if args.use_synthetic:
        logger.info("Generating synthetic data...")
        X, y, rooms = data_loader.create_synthetic_data(n_samples=1000, n_timestamps=20, n_aps=14)
    else:
        try:
            X_train, y_train, rooms_train = data_loader.load_training_data()
            X_test, y_test, rooms_test = data_loader.load_test_data()
            logger.info("Loaded real dataset")
        except FileNotFoundError:
            logger.warning("Real data not found, using synthetic data")
            X, y, rooms = data_loader.create_synthetic_data(n_samples=1000)
            args.use_synthetic = True

    if args.use_synthetic:
        np.random.seed(config['training']['random_state'])
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        train_idx, test_idx = indices[:split], indices[split:]
        X_train = X[train_idx]
        y_train = y[train_idx]
        rooms_train = rooms[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        rooms_test = rooms[test_idx]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    logger.info("Preprocessing data...")
    preprocessor = RSSIPreprocessor()
    X_train = preprocessor.preprocess(X_train, fit=True)
    X_test = preprocessor.preprocess(X_test, fit=False)

    logger.info("Training HMM-DMD model...")
    model = HMMDMDLocalizer(
        dmd_rank=config['model']['hmm_dmd']['dmd_rank'],
        n_components=config['model']['hmm_dmd']['n_components'],
        n_mix=config['model']['hmm_dmd']['n_mix'],
        cov_type=config['model']['hmm_dmd']['cov_type']
    )
    model.fit(X_train, y_train, rooms_train)
    logger.info("Model training completed")

    logger.info("Evaluating model...")
    predictions = model.predict(X_test)
    evaluator = LocalizationEvaluator()
    results = evaluator.comprehensive_evaluate(y_test, predictions)
    evaluator.print_results(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    logger.info("Generating visualizations...")
    plot_results(y_test, predictions, save_path=str(output_dir / 'localization_results.png'))
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Mean localization error: {results['mean_error']:.2f} m")


if __name__ == '__main__':
    main()

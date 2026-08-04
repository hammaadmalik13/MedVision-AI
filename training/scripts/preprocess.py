"""Preprocess BraTS dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from training.pipelines.preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BraTS Dataset")
    parser.add_argument("--input", type=str, default="./datasets/brats")
    parser.add_argument("--output", type=str, default="./data/cache")
    parser.add_argument("--config", type=str, default="configs/data/brats.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = PreprocessingPipeline(config)
    cached = pipeline.process_dataset(Path(args.input), Path(args.output))
    logger.info("Preprocessed %d subjects to %s", len(cached), args.output)


if __name__ == "__main__":
    main()

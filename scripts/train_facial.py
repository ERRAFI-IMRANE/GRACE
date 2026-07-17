"""
Training entry point for the DSCT facial stream.

Unlike the skeleton stream, the provided DSCT source file
(dsct_complete_pipeline.py) contains only an inference pipeline
(`DSCTFacialStream.predict`) built around a pretrained checkpoint - it
does not include the DSCT model's training loop, loss functions, or
matcher logic. That training logic lives in the external DSCT codebase
itself (see `third_party/DSCT/`).

This script is intentionally left as a thin pointer rather than a
fabricated training loop, since writing one without the original
matcher/loss implementation would not accurately reproduce DSCT
training and could silently produce incorrect results.

To train or fine-tune the DSCT facial stream:
    1. Place your local DSCT codebase under `third_party/DSCT/`.
    2. Follow that repository's own training script and instructions
       (typically a `main.py` or `train.py` at its root) to produce a
       checkpoint compatible with `DSCTFacialStream`.
    3. Point `configs/facial_config.yaml` -> `paths.ckpt_path` at the
       resulting checkpoint.

Usage (inference only, once a checkpoint is available):
    python scripts/evaluate.py --stream facial --config configs/facial_config.yaml
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Placeholder for DSCT facial stream training. See module docstring."
    )
    parser.add_argument("--config", type=str, default="configs/facial_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "Training logic for the DSCT facial stream is not included in this "
        "repository. See this file's module docstring for how to train via "
        "the external DSCT codebase under third_party/DSCT/."
    )


if __name__ == "__main__":
    main()

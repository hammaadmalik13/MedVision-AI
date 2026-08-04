"""BraTS dataset download helper."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_structure(output_dir: Path, num_samples: int = 3) -> None:
    """Create synthetic sample data for development and testing."""
    import nibabel as nib
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        subject_id = f"BraTS-GLI-0000{i}-000"
        subject_dir = output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)

        shape = (240, 240, 155)
        affine = np.eye(4)
        for modality in ["t1", "t1ce", "t2", "flair"]:
            data = np.random.randn(*shape).astype(np.float32) * 100 + 500
            img = nib.Nifti1Image(data, affine)
            nib.save(img, str(subject_dir / f"{subject_id}_{modality}.nii.gz"))

        seg = np.zeros(shape, dtype=np.uint8)
        center = (120, 120, 77)
        seg[center[0] - 20 : center[0] + 20, center[1] - 20 : center[1] + 20, center[2] - 10 : center[2] + 10] = 2
        seg[center[0] - 10 : center[0] + 10, center[1] - 10 : center[1] + 10, center[2] - 5 : center[2] + 5] = 1
        seg[center[0] - 5 : center[0] + 5, center[1] - 5 : center[1] + 5, center[2] - 3 : center[2] + 3] = 4
        nib.save(nib.Nifti1Image(seg, affine), str(subject_dir / f"{subject_id}_seg.nii.gz"))

    manifest = {"subjects": [f"BraTS-GLI-0000{i}-000" for i in range(num_samples)], "synthetic": True}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Created %d synthetic samples in %s", num_samples, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/Setup BraTS Dataset")
    parser.add_argument("--output", type=str, default="./datasets/brats")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic samples for dev")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.synthetic:
        create_sample_structure(output_dir, args.num_samples)
        return

    logger.info(
        "BraTS dataset requires registration at https://www.synapse.org/#!Synapse:syn27046444\n"
        "After download, extract to: %s\n"
        "Expected structure: subject_id/subject_id_modality.nii.gz",
        output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = output_dir / "README.md"
    readme.write_text(
        "# BraTS Dataset\n\n"
        "1. Register at https://www.synapse.org/#!Synapse:syn27046444\n"
        "2. Download BraTS GLI training data\n"
        "3. Extract to this directory\n"
        "4. Run: python -m training.scripts.preprocess\n\n"
        "For development without real data:\n"
        "`python -m training.scripts.download_brats --synthetic`\n"
    )


if __name__ == "__main__":
    main()

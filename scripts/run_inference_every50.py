import os
import os.path as osp
import sys
import shutil
import subprocess
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from src.model.detector import NOCTIS


# ============================================================
# SETTINGS
# ============================================================

EVERY_N = 50

RAW_IMAGE_DIR = "datasets/raw_tetris/extracted_tetris"

SAMPLED_DATASET_DIR = "datasets/tetris_every50"

VISUALIZATION_DIR = "datasets/results/noctis/every50"

# Existing Tetris templates
TEMPLATE_DIR = "datasets/templates/tetris"


# ============================================================
# BUILD SMALL BOP-STYLE DATASET
# ============================================================

def build_sampled_dataset():
    raw_dir = Path(RAW_IMAGE_DIR)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw image directory does not exist: {raw_dir}"
        )

    image_paths = sorted(
        raw_dir.glob("*.png"),
        key=lambda p: int(p.stem)
    )

    if not image_paths:
        raise RuntimeError(
            f"No PNG images found in {raw_dir}"
        )

    # Every 50th image: 0, 50, 100, ...
    selected_paths = image_paths[::EVERY_N]

    print("=" * 70)
    print("RAW DATASET")
    print("=" * 70)
    print(f"Total raw images:      {len(image_paths)}")
    print(f"Sampling every:        {EVERY_N}")
    print(f"Selected images:       {len(selected_paths)}")
    print()

    sampled_rgb_dir = Path(
        SAMPLED_DATASET_DIR
    ) / "test" / "000001" / "rgb"

    sampled_rgb_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove old sampled images so reruns are clean.
    for old_file in sampled_rgb_dir.glob("*.png"):
        old_file.unlink()

    selected_frame_ids = []

    for src in selected_paths:
        frame_id = int(src.stem)
        dst = sampled_rgb_dir / f"{frame_id:06d}.png"

        # Copy rather than symlink so the sampled dataset is
        # independent of the raw dataset.
        shutil.copy2(src, dst)

        selected_frame_ids.append(frame_id)

    # Copy the known camera calibration.
    source_camera = Path(
        "datasets/tetris/test/000001/scene_camera.json"
    )

    if not source_camera.exists():
        raise FileNotFoundError(
            f"Missing camera file: {source_camera}"
        )

    with open(source_camera, "r") as f:
        import json
        camera = json.load(f)

    # The original test scene only contains calibration for frame 0.
    # For this experiment we assume the camera calibration is constant
    # across the extracted frames, so reuse it for every sampled frame.
    if "000000" in camera:
        camera_000000 = camera["000000"]
    elif 0 in camera:
        camera_000000 = camera[0]
    else:
        # Fallback: use the first available camera entry.
        camera_000000 = next(iter(camera.values()))

    sampled_camera = {
        str(frame_id).zfill(6): camera_000000
        for frame_id in selected_frame_ids
    }

    sampled_camera_path = (
        Path(SAMPLED_DATASET_DIR)
        / "test"
        / "000001"
        / "scene_camera.json"
    )

    with open(sampled_camera_path, "w") as f:
        json.dump(
            sampled_camera,
            f,
            indent=2
        )

    print(f"Sampled dataset created at:")
    print(f"  {SAMPLED_DATASET_DIR}")
    print()

    print("Selected frame IDs:")
    print(
        "  " +
        ", ".join(
            f"{x:06d}" for x in selected_frame_ids
        )
    )
    print()

    return selected_frame_ids


# ============================================================
# MAIN
# ============================================================

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run_inference"
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    logging.basicConfig(
        level=logging.INFO
    )

    # --------------------------------------------------------
    # Build sampled dataset
    # --------------------------------------------------------

    selected_frame_ids = build_sampled_dataset()

    # --------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------

    logging.info(
        "Initializing trainer"
    )

    trainer = instantiate(
        cfg.machine.trainer
    )

    # --------------------------------------------------------
    # QUERY DATASET
    # --------------------------------------------------------

    query_config = cfg.data.query_dataloader.copy()

    query_config.root_dir = (
        SAMPLED_DATASET_DIR
    )

    query_config.split = "test"

    query_dataset = instantiate(
        query_config
    )

    logging.info(
        "Loaded sampled query dataset with %d images",
        len(query_dataset)
    )

    # --------------------------------------------------------
    # REFERENCE / TEMPLATE DATASET
    # --------------------------------------------------------

    ref_config = (
        cfg.data.reference_dataloader.copy()
    )

    ref_config.template_dir = (
        TEMPLATE_DIR
    )

    ref_dataset = instantiate(
        ref_config
    )

    logging.info(
        "Loaded Tetris reference templates"
    )

    # --------------------------------------------------------
    # NOCTIS MODEL
    # --------------------------------------------------------

    logging.info(
        "Initializing NOCTIS model"
    )

    model = instantiate(
        cfg.model
    )

    model.dataset_name = "tetris"
    model.ref_dataset = ref_dataset

    # --------------------------------------------------------
    # Output root
    # --------------------------------------------------------

    os.makedirs(
        VISUALIZATION_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process one image at a time
    # --------------------------------------------------------

    for sample_number, frame_id in enumerate(
        selected_frame_ids,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"SAMPLE {sample_number}/{len(selected_frame_ids)}"
        )
        print(
            f"FRAME: {frame_id:06d}"
        )
        print("=" * 70)

        # Find this frame inside the BOP metadata.
        matching_indices = [
            i
            for i in range(len(query_dataset))
            if int(
                query_dataset.metadata.iloc[i]["frame_id"]
            ) == frame_id
        ]

        if len(matching_indices) != 1:
            raise RuntimeError(
                f"Could not uniquely find frame {frame_id:06d}"
            )

        dataset_index = matching_indices[0]

        # One-image subset.
        one_image_dataset = Subset(
            query_dataset,
            [dataset_index]
        )

        query_loader = DataLoader(
            one_image_dataset,
            batch_size=1,
            num_workers=cfg.machine.num_workers,
            shuffle=False
        )

        # Unique name so every sample gets its own JSON.
        prediction_name = (
            f"result_tetris_every50_frame_{frame_id:06d}"
        )

        # Remove stale result directory/json from a previous run.
        prediction_dir = Path(
            "datasets/results/noctis"
        ) / prediction_name

        prediction_json = (
            Path("datasets/results/noctis")
            / f"{prediction_name}.json"
        )

        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)

        if prediction_json.exists():
            prediction_json.unlink()

        model.name_prediction_file = prediction_name

        # ----------------------------------------------------
        # NOCTIS INFERENCE
        # ----------------------------------------------------

        logging.info(
            "Running NOCTIS on frame %06d",
            frame_id
        )

        trainer.test(
            model,
            dataloaders=query_loader
        )

        # ----------------------------------------------------
        # FIND RESULT JSON
        # ----------------------------------------------------

        if not prediction_json.exists():
            raise RuntimeError(
                f"NOCTIS did not create expected JSON:\n"
                f"{prediction_json}"
            )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        frame_visualization_dir = (
            Path(VISUALIZATION_DIR)
            / f"{frame_id:06d}"
        )

        frame_visualization_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.info(
            "Visualizing frame %06d",
            frame_id
        )

        visualize_command = [
            sys.executable,
            "scripts/visualize_detectron2.py",

            "-i",
            str(prediction_json),

            "-d",
            SAMPLED_DATASET_DIR,

            "-s",
            "test",

            "-u",
            "rgb",

            "-b",
            "rgb",

            "-o",
            str(frame_visualization_dir)
        ]

        subprocess.run(
            visualize_command,
            check=True
        )

        print()
        print(
            f"Finished frame {frame_id:06d}"
        )
        print(
            f"Visualization directory:"
        )
        print(
            f"  {frame_visualization_dir}"
        )

    print()
    print("=" * 70)
    print("ALL SAMPLED IMAGES FINISHED")
    print("=" * 70)
    print()
    print(
        f"Visualizations are in:"
    )
    print(
        f"  {VISUALIZATION_DIR}"
    )


if __name__ == "__main__":
    main()

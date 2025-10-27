"""Utility script to run inference with the 15-channel EMG model.

This entry point loads a trained checkpoint, fetches samples from the
specified dataset split, and prints the predicted label alongside the
ground-truth target.  It can also export predictions for the entire
split to a CSV file so results can be inspected offline.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from emg15_dataset import EMGDataset15, MFSCConfig
from emg15_model import EMGNet15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect predictions from a trained 15-channel EMG model")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint produced by emg15_main.py")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--sample-index", type=int, default=None, help="Optional index of a single sample to inspect")
    parser.add_argument("--output-csv", type=str, default="", help="Optional path to export predictions for the entire split")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size when exporting the whole split")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers when exporting")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computation device preference")
    parser.add_argument("--topk", type=int, default=5, help="How many highest-probability classes to display for a single sample")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_checkpoint(path: str, device: torch.device) -> Tuple[EMGNet15, Dict]:
    checkpoint = torch.load(path, map_location="cpu")
    args: Dict = checkpoint.get("args", {})

    num_classes = args.get("num_classes")
    if num_classes is None:
        classifier_weight = checkpoint["model"]["classifier.1.weight"]
        num_classes = classifier_weight.shape[0]

    model = EMGNet15(
        num_classes=num_classes,
        in_channels=args.get("in_channels", 15),
        proj_dim=args.get("proj_dim", 256),
        gru_hidden=args.get("gru_hidden", 512),
        gru_layers=args.get("gru_layers", 2),
        gru_dropout=args.get("gru_dropout", 0.3),
        every_frame=args.get("every_frame", True),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def reduce_outputs(model: EMGNet15, outputs: torch.Tensor) -> torch.Tensor:
    if model.every_frame and outputs.ndim == 3:
        return outputs.mean(dim=1)
    return outputs


def run_single_sample(
    model: EMGNet15,
    dataset: EMGDataset15,
    index: int,
    device: torch.device,
    *,
    topk: int = 5,
) -> None:
    if not 0 <= index < len(dataset):
        raise IndexError(f"Sample index {index} is out of bounds for split of size {len(dataset)}")

    label_name, file_path = dataset.file_list[index]
    features, target = dataset[index]
    inputs = features.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = reduce_outputs(model, model(inputs))
        probabilities = torch.softmax(logits, dim=1)
        prediction = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, prediction])

    print("--- Single-sample inspection ---", flush=True)
    print(f"File: {file_path}", flush=True)
    print(f"Ground truth (label id / name): {target} / {label_name}", flush=True)
    print(f"Predicted class: {prediction} (confidence {confidence:.4f})", flush=True)

    if topk > 1:
        topk = min(topk, probabilities.shape[1])
        confs, indices = torch.topk(probabilities, k=topk, dim=1)
        print("Top predictions:")
        for rank in range(topk):
            idx = int(indices[0, rank].item())
            conf = float(confs[0, rank].item())
            print(f"  #{rank + 1}: class {idx} | confidence {conf:.4f}")


def iterate_dataset(
    model: EMGNet15,
    dataset: EMGDataset15,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Iterable[Tuple[int, int, int, float]]:
    model.eval()
    sample_index = 0
    with torch.no_grad():
        for inputs, targets in loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = reduce_outputs(model, model(inputs))
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)

            for i in range(batch_size):
                prob = float(probabilities[i, preds[i]].item())
                yield sample_index, int(targets[i].item()), int(preds[i].item()), prob
                sample_index += 1


def export_predictions(
    model: EMGNet15,
    dataset: EMGDataset15,
    loader: torch.utils.data.DataLoader,
    csv_path: Path,
    device: torch.device,
) -> Tuple[int, int]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label_id", "label_name", "prediction", "confidence", "correct", "file_path"])

        for sample_idx, target, pred, confidence in iterate_dataset(model, dataset, loader, device):
            label_name, file_path = dataset.file_list[sample_idx]
            is_correct = int(pred == target)
            writer.writerow([sample_idx, target, label_name, pred, f"{confidence:.6f}", is_correct, file_path])

            correct += is_correct
            total += 1

    return correct, total


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    cfg = MFSCConfig()
    dataset = EMGDataset15(args.dataset_root, split=args.split, cfg=cfg)
    print(f"Loaded {args.split} split with {len(dataset)} samples from {args.dataset_root}", flush=True)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    epoch = checkpoint.get("epoch")
    epoch_info = f" (epoch {epoch})" if epoch is not None else ""
    print(f"Loaded checkpoint from {args.checkpoint}{epoch_info}", flush=True)

    if args.sample_index is None and not args.output_csv:
        raise SystemExit("Please specify --sample-index to inspect an example or --output-csv to export the entire split.")

    if args.sample_index is not None:
        run_single_sample(model, dataset, args.sample_index, device, topk=args.topk)

    if args.output_csv:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        csv_path = Path(args.output_csv)
        correct, total = export_predictions(model, dataset, loader, csv_path, device)
        accuracy = correct / total if total else 0.0
        print(
            f"Saved predictions for {total} samples to {csv_path} | accuracy {accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

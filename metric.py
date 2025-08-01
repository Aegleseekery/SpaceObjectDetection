import os
import math
import detect
from typing import List, Tuple, Optional, Union
import numpy as np
import json
import csv
from scipy.spatial import cKDTree


def load_yolo_annotation(yolo_path: str, image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """
    Load YOLO-format annotation and return list of centroids (x, y) in pixel coordinates.
    YOLO format per line: <class> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    image_shape: (height, width)
    """
    h, w = image_shape
    centers = []
    if not os.path.exists(yolo_path):
        return centers
    with open(yolo_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            _, x_c_norm, y_c_norm, bw_norm, bh_norm = map(float, parts[:5])
            x = x_c_norm * w - 1
            y = y_c_norm * h - 1
            centers.append((x, y))
    return centers


def load_detector_txt(txt_path: str) -> List[Tuple[float, float]]:
    """
    Load detector output txt file expecting lines like:
    cx cy area w h gray_sum
    Returns list of (cx, cy).
    """
    centers = []
    if not os.path.exists(txt_path):
        return centers
    with open(txt_path, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            cx = float(parts[0])
            cy = float(parts[1])
            centers.append((cx, cy))
    return centers


def compute_point_metrics(pred_points: List[Tuple[float, float]],
                          gt_points: List[Tuple[float, float]],
                          R: float) -> Tuple[int, int, int, float]:
    """
    Given predicted centroids and ground truth centroids, compute TP, FP, FN and average localization error (ALE).
    Matching is greedy nearest neighbor with threshold R, no duplicate GT matching.
    Returns: TP, FP, FN, average_distance
    """
    if len(pred_points) == 0:
        return 0, 0, len(gt_points), 0.0
    if len(gt_points) == 0:
        return 0, len(pred_points), 0, 0.0

    gt_tree = cKDTree(gt_points)
    matched_gt = set()
    distances = []
    TP = 0
    FP = 0

    for px, py in pred_points:
        dist, idx = gt_tree.query((px, py), k=1)
        if dist <= R and idx not in matched_gt:
            TP += 1
            matched_gt.add(idx)
            distances.append(dist)
        else:
            FP += 1

    FN = len(gt_points) - len(matched_gt)
    avg_dist = float(np.mean(distances)) if distances else 0.0
    return TP, FP, FN, avg_dist


class Evaluation:
    def __init__(self,
                 detector_output_dir: str,
                 gt_label_dir: str,
                 save_dir: str,
                 image_shape=(256,256),
                 match_radius: float = 1.5):
        """
        :param detector_output_dir: directory containing detector .txt outputs (named like <basename>_<method>.txt)
        :param gt_label_dir: directory containing YOLO-format ground truth files (same basename, e.g., 0001.txt)
        :param save_dir: directory to dump evaluation summaries
        :param image_size_map: optional dict mapping basename -> (height, width); needed to interpret YOLO coords
        :param match_radius: pixel threshold for matching centroids
        """
        self.pred_dir = detector_output_dir
        self.gt_dir = gt_label_dir
        self.save_dir = save_dir
        self.image_shape = image_shape
        self.R = match_radius

    def evaluate_pair(self,
                      pred_path: str,
                      gt_path: str,
                      image_shape: Tuple[int, int]) -> Tuple[int, int, int, float]:
        """
        Evaluate one pair of files; returns TP, FP, FN, avg_dist.
        """
        pred_points = load_detector_txt(pred_path)
        gt_points = load_yolo_annotation(gt_path, image_shape)
        TP, FP, FN, avg_dist = compute_point_metrics(pred_points, gt_points, self.R)
        return TP, FP, FN, avg_dist

    def run(self) -> dict:
        """
        Runs evaluation over all matching base filenames in pred_dir and gt_dir.
        Returns summary dictionary with aggregated metrics and per-file breakdown.
        Also saves results to save_dir if provided.
        """
        summary = {
            "files": [],
            "total_TP": 0,
            "total_FP": 0,
            "total_FN": 0,
            "avg_distance_list": []
        }

        for pred_fname in os.listdir(self.pred_dir):
            if not pred_fname.lower().endswith(".txt"):
                continue
            basename = os.path.splitext(pred_fname)[0]
            if "_" in basename:
                gt_basename = basename.rsplit("_", 1)[0]
            else:
                gt_basename = basename
            pred_path = os.path.join(self.pred_dir, pred_fname)
            gt_path = os.path.join(self.gt_dir, f"{gt_basename}.txt")
            TP, FP, FN, avg_dist = self.evaluate_pair(pred_path, gt_path, self.image_shape)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            summary["files"].append({
                "file": pred_fname,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "avg_distance": avg_dist,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            summary["total_TP"] += TP
            summary["total_FP"] += FP
            summary["total_FN"] += FN
            if avg_dist > 0:
                summary["avg_distance_list"].append(avg_dist)

        # aggregate overall metrics
        total_TP = summary["total_TP"]
        total_FP = summary["total_FP"]
        total_FN = summary["total_FN"]
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_distance = float(np.mean(summary["avg_distance_list"])) if summary["avg_distance_list"] else 0.0

        summary["precision"] = precision
        summary["recall"] = recall
        summary["f1"] = f1
        summary["avg_distance"] = avg_distance

        # Save summary and per-file breakdown if requested
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            # JSON overall summary
            summary_path = os.path.join(self.save_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "avg_distance": avg_distance,
                    "total_TP": total_TP,
                    "total_FP": total_FP,
                    "total_FN": total_FN,
                }, f, indent=2)

            # CSV per-file
            perfile_path = os.path.join(self.save_dir, "per_file.csv")
            with open(perfile_path, "w", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "file", "TP", "FP", "FN", "precision", "recall", "f1", "avg_distance"
                ])
                writer.writeheader()
                for row in summary["files"]:
                    writer.writerow(row)

        return summary


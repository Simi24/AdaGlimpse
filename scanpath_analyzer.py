from architectures.rl.actor_critic import ActorCritic
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from dtaidistance import dtw
from sklearn.metrics import pairwise_distances
import torch
import json
from pathlib import Path


class ScanpathAnalyzer:
    def __init__(self, image_width=224, image_height=224):
        self.image_width = image_width
        self.image_height = image_height
        self.model_scanpaths = {}
        self.human_scanpaths = {}

    def load_coco_eyetracking(self, data_path):
        """Load COCO Free Viewing dataset with eye tracking data"""
        with open(data_path, "r") as f:
            self.human_scanpaths = json.load(f)

    def track_model_scanpath(self, image_id, coordinates, timestamp):
        """Track model-generated coordinates for an image"""
        if image_id not in self.model_scanpaths:
            self.model_scanpaths[image_id] = []

        # Convert normalized coordinates to pixel space
        pixel_coords = {
            "x": coordinates[0] * self.image_width,
            "y": coordinates[1] * self.image_height,
            "timestamp": timestamp,
        }
        self.model_scanpaths[image_id].append(pixel_coords)

    def compute_metrics(self, image_id):
        """Compute comparison metrics between model and human scanpaths"""
        model_path = self._get_coordinate_sequence(self.model_scanpaths[image_id])
        human_path = self._get_coordinate_sequence(self.human_scanpaths[image_id])

        return {
            "dtw_distance": self._compute_dtw(model_path, human_path),
            "sequence_score": self._compute_sequence_score(model_path, human_path),
            "attention_similarity": self._compute_attention_similarity(model_path, human_path),
            "temporal_correlation": self._compute_temporal_correlation(model_path, human_path),
        }

    def _get_coordinate_sequence(self, scanpath):
        """Convert scanpath dictionary to numpy array of coordinates"""
        return np.array([[p["x"], p["y"]] for p in scanpath])

    def _compute_dtw(self, path1, path2):
        """Compute Dynamic Time Warping distance between scanpaths"""
        return dtw.distance(path1, path2)

    def _compute_sequence_score(self, path1, path2):
        """Compute sequence similarity score based on order of fixations"""
        distances = pairwise_distances(path1, path2)
        return np.mean(np.min(distances, axis=1))

    def _compute_attention_similarity(self, path1, path2):
        """Compute similarity of attention distribution"""

        def create_heatmap(path, size=(32, 32)):
            heatmap = np.zeros(size)
            for x, y in path:
                x_bin = int(x * size[0] / self.image_width)
                y_bin = int(y * size[1] / self.image_height)
                heatmap[y_bin, x_bin] += 1
            return heatmap / np.sum(heatmap)

        heatmap1 = create_heatmap(path1)
        heatmap2 = create_heatmap(path2)
        return 1 - entropy(heatmap1.flatten(), heatmap2.flatten())

    def _compute_temporal_correlation(self, path1, path2):
        """Compute temporal correlation of scanpaths"""
        min_len = min(len(path1), len(path2))
        path1 = path1[:min_len]
        path2 = path2[:min_len]

        velocities1 = np.diff(path1, axis=0)
        velocities2 = np.diff(path2, axis=0)

        correlation = np.corrcoef(velocities1.flatten(), velocities2.flatten())[0, 1]
        return correlation


class ScanpathAwareActorCritic(ActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scanpath_analyzer = ScanpathAnalyzer()
        self.current_step = 0

    def forward(self, tensordict):
        output = self.policy_module(tensordict)

        # Track coordinates
        if "image_id" in tensordict:
            image_id = tensordict["image_id"]
            loc = output.get("loc")
            self.scanpath_analyzer.track_model_scanpath(
                image_id=image_id, coordinates=loc.detach().cpu().numpy(), timestamp=self.current_step
            )
            self.current_step += 1

        return output

    def analyze_scanpath(self, image_id):
        """Analyze scanpath for a specific image"""
        return self.scanpath_analyzer.compute_metrics(image_id)

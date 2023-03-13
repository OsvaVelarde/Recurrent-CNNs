'''
References:
'''

# ===========================================================
import numpy as np

from torch.utils.data.sampler import BatchSampler, Sampler
from torchvision.datasets import CocoDetection as visionCoco
from torch.utils.data import Subset

import bisect
from collections import defaultdict
import copy

# =============================================================

class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size

# =============================================================

def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)

def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, visionCoco):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    #return _compute_aspect_ratios_slow(dataset, indices)

def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))
    return groups

# =============================================================
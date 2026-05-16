import torch
from torch.utils.data import Sampler
import random
import math
from typing import List, Dict, Any

class BatchRatioSampler(Sampler):
    def __init__(self, samples: List[Dict[str, Any]], batch_size: int, ratio: str = "1:1"):
        self.samples = samples
        self.batch_size = batch_size
        
        # Parse ratio
        try:
            pos_part, neg_part = map(int, ratio.split(':'))
            if pos_part <= 0 or neg_part <= 0:
                raise ValueError
            self.pos_ratio = pos_part / (pos_part + neg_part)
        except:
            raise ValueError(f"ratio must look like '1:1' with positive integers, got '{ratio}'")
            
        self.pos_indices = [i for i, s in enumerate(samples) if s['role'] == 'positive']
        self.neg_indices = [i for i, s in enumerate(samples) if s['role'] in ['negative', 'synthetic']]

        has_pos = len(self.pos_indices) > 0
        has_neg = len(self.neg_indices) > 0
        if has_pos and has_neg:
            self.num_pos_per_batch = max(1, min(self.batch_size - 1, int(self.batch_size * self.pos_ratio)))
            self.num_neg_per_batch = self.batch_size - self.num_pos_per_batch
        elif has_pos:
            self.num_pos_per_batch = self.batch_size
            self.num_neg_per_batch = 0
        elif has_neg:
            self.num_pos_per_batch = 0
            self.num_neg_per_batch = self.batch_size
        else:
            self.num_pos_per_batch = 0
            self.num_neg_per_batch = 0

        pos_batches = math.ceil(len(self.pos_indices) / self.num_pos_per_batch) if self.num_pos_per_batch else 0
        neg_batches = math.ceil(len(self.neg_indices) / self.num_neg_per_batch) if self.num_neg_per_batch else 0
        self.num_batches = max(pos_batches, neg_batches)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            
            # Sample with replacement so scarce positives/negatives can still satisfy the batch ratio.
            if self.num_pos_per_batch > 0:
                batch.extend(random.choices(self.pos_indices, k=self.num_pos_per_batch))
            
            if self.num_neg_per_batch > 0:
                batch.extend(random.choices(self.neg_indices, k=self.num_neg_per_batch))
            
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.num_batches * self.batch_size

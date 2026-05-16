import torch
from torch.utils.data import Sampler
import random
from typing import List, Dict, Any

class BatchRatioSampler(Sampler):
    def __init__(self, samples: List[Dict[str, Any]], batch_size: int, ratio: str = "1:2"):
        self.samples = samples
        self.batch_size = batch_size
        
        # Parse ratio
        try:
            pos_part, neg_part = map(int, ratio.split(':'))
            self.pos_ratio = pos_part / (pos_part + neg_part)
        except:
            self.pos_ratio = 0.33 # Default 1:2
            
        self.pos_indices = [i for i, s in enumerate(samples) if s['role'] == 'positive']
        self.neg_indices = [i for i, s in enumerate(samples) if s['role'] in ['negative', 'synthetic']]
        
        self.num_pos_per_batch = int(self.batch_size * self.pos_ratio)
        self.num_neg_per_batch = self.batch_size - self.num_pos_per_batch
        
        # Total number of batches based on the larger pool (or smaller? usually larger with oversampling)
        # Requirement says "Oversamples positives or subsamples negatives as needed"
        # Let's target the larger pool to avoid missing data, or just use a fixed number.
        # Typically, we want to see all positives at least once.
        self.num_batches = max(len(self.pos_indices) // self.num_pos_per_batch, 
                               len(self.neg_indices) // self.num_neg_per_batch)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            
            # Sample positives
            if len(self.pos_indices) > 0:
                batch.extend(random.choices(self.pos_indices, k=self.num_pos_per_batch))
            
            # Sample negatives
            if len(self.neg_indices) > 0:
                batch.extend(random.choices(self.neg_indices, k=self.num_neg_per_batch))
            
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.num_batches * self.batch_size

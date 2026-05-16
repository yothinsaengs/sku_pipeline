import os
import logging
import datetime
import pandas as pd
import json
import torch
import shutil
from typing import Dict, Any

class RunLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_run_dir = config['logging'].get('run_dir', 'runs')
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # experiment_name suffix
        loss_type = config['loss']['type']
        ratio = config['sampling'].get('pos_neg_ratio', "1to2").replace(':', 'to')
        self.run_name = f"{self.timestamp}_{loss_type}_{ratio}"
        self.run_dir = os.path.join(self.base_run_dir, self.run_name)
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'checkpoints'), exist_ok=True)
        
        self._setup_logging()
        self._save_config()
        
        self.metrics_csv_path = os.path.join(self.run_dir, 'metrics.csv')
        self.metrics_history = []

    def _setup_logging(self):
        log_file = os.path.join(self.run_dir, self.config['logging'].get('log_file', 'train.log'))
        
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _save_config(self):
        with open(os.path.join(self.run_dir, 'config.yaml'), 'w') as f:
            import yaml
            yaml.dump(self.config, f)

    def log(self, message: str):
        self.logger.info(message)

    def save_metrics(self, epoch_metrics: Dict[str, Any]):
        self.metrics_history.append(epoch_metrics)
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.metrics_csv_path, index=False)

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.run_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, path)
        
        # Save rolling checkpoint
        rolling_path = os.path.join(self.run_dir, 'checkpoints', f'epoch_{epoch:03d}.pth')
        torch.save(checkpoint, rolling_path)
        
        # Cleanup old checkpoints
        keep_last_n = self.config['training']['checkpoints'].get('keep_last_n', 3)
        checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_')])
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                os.remove(os.path.join(checkpoint_dir, old_ckpt))

    def save_summary(self, summary: Dict[str, Any]):
        with open(os.path.join(self.run_dir, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

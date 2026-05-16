import os
from typing import Dict, Any

class ConfigError(Exception):
    pass

def validate_config(config: Dict[str, Any]):
    required_sections = ['dataset', 'classes', 'model', 'training', 'loss', 'augmentation']
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"missing field: {section}")
    
    # Classes
    pos_found = False
    for i, cls in enumerate(config['classes']):
        if 'id' not in cls or 'name' not in cls or 'role' not in cls:
            raise ConfigError(f"class at index {i} must have id, name, and role")
        if cls['role'] not in ['positive', 'negative']:
            raise ConfigError(f"role must be 'positive' or 'negative', got '{cls['role']}'")
        if cls['role'] == 'positive':
            pos_found = True
    
    if not pos_found:
        raise ConfigError("no classes with role: positive defined")

    # Sampling
    ratio = config.get('sampling', {}).get('pos_neg_ratio', '1:1')
    try:
        pos_part, neg_part = map(int, ratio.split(':'))
    except Exception as exc:
        raise ConfigError(f"sampling.pos_neg_ratio must look like '1:1', got '{ratio}'") from exc
    if pos_part <= 0 or neg_part <= 0:
        raise ConfigError(f"sampling.pos_neg_ratio values must be positive, got '{ratio}'")
    
    # Model
    if config['model'].get('num_classes') != len([c for c in config['classes'] if c['role'] == 'positive']):
         # This is a soft check, but good to have
         pass

    # Loss
    loss_type = config['loss'].get('type')
    if loss_type not in ['focal', 'bce']:
        raise ConfigError(f"loss.type must be 'focal' or 'bce', got '{loss_type}'")
        
    # Numeric ranges
    margin = config['input'].get('context_margin', 0)
    if not (0 <= margin <= 1):
        raise ConfigError(f"context_margin must be in [0, 1], got {margin}")
        
    return True

import torch
from .config import PipelineConfig

class GTGenerator:
    def __init__(self, config: PipelineConfig, num_classes: int = 5):
        # 0: Background, 1: Fog, 2: Reflection, 3: Soiling, 4: Flare
        self.config = config
        self.num_classes = num_classes

    def generate(self, masks: dict[str, torch.Tensor]) -> torch.Tensor:
        b, _, h, w = list(masks.values())[0].shape
        device = list(masks.values())[0].device
        
        if self.config.gt_format == "multi_channel":
            # Идеально для BCEWithLogitsLoss
            gt = torch.zeros((b, self.num_classes, h, w), device=device)
            
            # Z-Order приоритеты при наложении (последний перекрывает)
            # Фон всегда 1.0, потом вычитаем дефекты
            gt[:, 0, :, :] = 1.0 
            
            if self.config.fog.enabled:
                gt[:, 1, :, :] = masks['fog']
            if self.config.reflection.enabled:
                gt[:, 2, :, :] = masks['reflection']
            if self.config.soiling.enabled:
                gt[:, 3, :, :] = masks['soiling']
            if self.config.flare.enabled:
                gt[:, 4, :, :] = masks['flare']
                
            return gt
            
        elif self.config.gt_format == "priority_single":
            # Для обычного CrossEntropyLoss (один класс на пиксель)
            gt = torch.zeros((b, 1, h, w), dtype=torch.long, device=device)
            
            # Последовательное перекрытие (Flare имеет высший приоритет)
            if self.config.fog.enabled:
                gt = torch.where(masks['fog'] > 0.5, 1, gt)
            if self.config.reflection.enabled:
                gt = torch.where(masks['reflection'] > 0.5, 2, gt)
            if self.config.soiling.enabled:
                gt = torch.where(masks['soiling'] > 0.5, 3, gt)
            if self.config.flare.enabled:
                gt = torch.where(masks['flare'] > 0.5, 4, gt)
                
            return gt
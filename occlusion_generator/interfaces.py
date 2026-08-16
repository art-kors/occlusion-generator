from abc import ABC, abstractmethod
import torch
from .config import ModuleConfig

class BaseOcclusionModule(ABC):
    """Базовый класс для всех генераторов окклюзий."""
    
    @abstractmethod
    def apply(self, image: torch.Tensor, depth: torch.Tensor, 
              car_mask: torch.Tensor, cfg: ModuleConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W) Текущее изображение
            depth: (B, 1, H, W) Карта глубины в метрах
            car_mask: (B, 1, H, W) Бинарная маска кузова
            cfg: Конфиг конкретного модуля
        Returns:
            tuple: (Измененное изображение, Бинарная/Float маска окклюзии)
        """
        pass
import torch
from .config import PipelineConfig
from .preprocessors import DepthEstimator, CarSegmentator
from .gt_generator import GTGenerator
from .modules.fog import FogModule
from .modules.reflection import ReflectionModule
from .modules.soiling import SoilingModule
from .modules.flare import FlareModule

class OcclusionPipeline:
    def __init__(self, config: PipelineConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Инициализация тяжелых моделей (один раз)
        self.depth_estimator = DepthEstimator(device=device)
        self.car_segmentator = CarSegmentator(device=device)
        
        # Инициализация легких модулей генерации
        self.modules = {
            'fog': FogModule(),
            'reflection': ReflectionModule(),
            'soiling': SoilingModule(),
            'flare': FlareModule()
        }
        
        self.gt_generator = GTGenerator(config)

    @torch.no_grad() # Отключаем градиенты для скорости
    def process(self, image: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W) чистый тензор изображения [0, 1]
            **kwargs: Дополнительные текстуры, например:
                     reflection_texture (B, 3, H, W) - тензор отражения
        Returns:
            tuple: (Occluded Image [0, 1], GT Masks)
        """
        image = image.to(self.device)
        b, c, h, w = image.shape
        
        # Шаг 0: Препроцессинг (только если нужен туман)
        depth = torch.ones((b, 1, h, w), device=self.device) * 10.0 # дефолтная дальняя глубина
        car_mask = torch.zeros((b, 1, h, w), device=self.device)
        
        if self.config.fog.enabled:
            depth = self.depth_estimator.estimate(image)
            car_mask = self.car_segmentator.segment(image)
            # Хак для боковых камер: зануляем глубину на кузове
            depth = torch.where(car_mask > 0.5, 0.3, depth)

        current_image = image.clone()
        generated_masks = {}

        # Шаг 1-4: Строго по Z-Order
        # 1. FOG (Меняет фон)
        current_image, mask = self.modules['fog'].apply(current_image, depth, car_mask, self.config.fog)
        generated_masks['fog'] = mask

        # 2. REFLECTION (На стекле) -> ПЕРЕДАЕМ **kwargs ЗДЕСЬ!
        current_image, mask = self.modules['reflection'].apply(
            current_image, depth, car_mask, self.config.reflection, **kwargs
        )
        generated_masks['reflection'] = mask

        # 3. SOILING (Поверх стекла)
        current_image, mask = self.modules['soiling'].apply(current_image, depth, car_mask, self.config.soiling)
        generated_masks['soiling'] = mask
        soil_mask_for_flare = mask # Передаем маску грязи в модуль бликов!

        # 4. FLARE (Внутри линзы, взаимодействует с грязью)
        current_image, mask = self.modules['flare'].apply(
            current_image, depth, car_mask, self.config.flare, soil_mask=soil_mask_for_flare
        )
        generated_masks['flare'] = mask

        # Финал: Сборка GT
        gt_masks = self.gt_generator.generate(generated_masks)
        
        # Ограничиваем пиксели картинки [0, 1] на случай аддитивного смешивания бликов
        current_image = torch.clamp(current_image, 0.0, 1.0)

        return current_image, gt_masks
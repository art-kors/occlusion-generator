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
        self.depth_estimator = DepthEstimator(device=device)
        self.car_segmentator = CarSegmentator(device=device)
        self.modules = {
            'fog': FogModule(),
            'reflection': ReflectionModule(),
            'soiling': SoilingModule(),
            'flare': FlareModule()
        }
        self.gt_generator = GTGenerator(config)

    @torch.no_grad()
    def process(self, image: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.to(self.device)
        b, c, h, w = image.shape
        depth = torch.ones((b, 1, h, w), device=self.device) * 10.0
        car_mask = torch.zeros((b, 1, h, w), device=self.device)
        
        if self.config.fog.enabled:
            depth = self.depth_estimator.estimate(image)
            car_mask = self.car_segmentator.segment(image)
            depth = torch.where(car_mask > 0.5, 0.3, depth)

        current_image = image.clone()
        generated_masks = {}

        current_image, mask = self.modules['fog'].apply(current_image, depth, car_mask, self.config.fog)
        generated_masks['fog'] = mask

        current_image, mask = self.modules['reflection'].apply(
            current_image, depth, car_mask, self.config.reflection, **kwargs
        )
        generated_masks['reflection'] = mask

        # ИСПРАВЛЕНИЕ ТУТ: ДОБАВЛЕН **kwargs !!!
        current_image, mask = self.modules['soiling'].apply(
            current_image, depth, car_mask, self.config.soiling, **kwargs
        )
        generated_masks['soiling'] = mask
        soil_mask_for_flare = mask 

        current_image, mask = self.modules['flare'].apply(
            current_image, depth, car_mask, self.config.flare, soil_mask=soil_mask_for_flare
        )
        generated_masks['flare'] = mask

        gt_masks = self.gt_generator.generate(generated_masks)
        current_image = torch.clamp(current_image, 0.0, 1.0)

        return current_image, gt_masks
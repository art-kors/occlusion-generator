from pydantic import BaseModel, Field
from typing import Tuple, Optional

class ModuleConfig(BaseModel):
    enabled: bool = False
    intensity: float = Field(0.0, ge=0.0, le=1.0, description="Сила эффекта от 0 до 1")

class FogConfig(ModuleConfig):
    color: Tuple[float, float, float] = (0.75, 0.78, 0.82) # RGB нормализованный

class ReflectionConfig(ModuleConfig):
    pass

class SoilingConfig(ModuleConfig):
    apply_distortion: bool = True # Преломление в каплях

class FlareConfig(ModuleConfig):
    pass

class PipelineConfig(BaseModel):
    fog: FogConfig = Field(default_factory=FogConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    soiling: SoilingConfig = Field(default_factory=SoilingConfig)
    flare: FlareConfig = Field(default_factory=FlareConfig)
    
    gt_format: str = "multi_channel" # или "priority_single"


class FogConfig(ModuleConfig):
    color: Tuple[float, float, float] = (0.75, 0.78, 0.82) # RGB нормализованный


class RainDropConfig(ModuleConfig):
    """
    Configuration for raindrop generation.

    Attributes:
        maxR: Maximum raindrop radius.
        minR: Minimum raindrop radius.
        maxDrops: Maximum number of raindrops in the image.
        minDrops: Minimum number of raindrops in the image.
        edge_darkratio: Brightness reduction factor for raindrop edges.
        return_label: Whether to return a label/mask along with the image.
        label_thres: Threshold used for generating the raindrop label.
        A: First Bezier control point in alpha-map radius coordinates.
        B: Second Bezier control point in alpha-map radius coordinates.
        C: Third Bezier control point in alpha-map radius coordinates.
        D: Fourth Bezier control point in alpha-map radius coordinates.
    """

    maxR: int = Field(default=100, le=150)
    minR: int = Field(default=1)

    maxDrops: int = Field(default=100)
    minDrops: int = Field(default=90)

    edge_darkratio: float = Field(default=0.6)

    return_label: bool = Field(default=True)
    label_thres: int = Field(default=128)

    A: tuple[float, float] = (1, 4.5)
    B: tuple[float, float] = (3, 1)
    C: tuple[float, float] = (1, 3)
    D: tuple[float, float] = (3, 3)
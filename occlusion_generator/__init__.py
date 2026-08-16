"""
OcclusionGenerator: A modular pipeline for generating realistic, 
multi-layered camera occlusions (Fog, Reflections, Soiling, Flare).
"""

from .config import (
    PipelineConfig,
    FogConfig,
    ReflectionConfig,
    SoilingConfig,
    FlareConfig
)
from .pipeline import OcclusionPipeline

__all__ = [
    "OcclusionPipeline",
    "PipelineConfig",
    "FogConfig",
    "ReflectionConfig",
    "SoilingConfig",
    "FlareConfig",
]
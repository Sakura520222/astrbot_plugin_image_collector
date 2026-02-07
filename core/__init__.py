"""
图片收集插件核心模块
"""

from core.config import PluginConfig
from core.exceptions import (
    ImageCollectorError,
    DownloadError,
    ProcessingError,
    HashCalculationError,
    CompressionError,
    FileOperationError,
    DuplicateImageError,
)

__all__ = [
    "PluginConfig",
    "ImageCollectorError",
    "DownloadError",
    "ProcessingError",
    "HashCalculationError",
    "CompressionError",
    "FileOperationError",
    "DuplicateImageError",
]
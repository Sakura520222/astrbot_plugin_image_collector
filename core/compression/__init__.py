# -*- coding: utf-8 -*-
"""
压缩策略模块 - 统一的图片压缩管理

提供统一的压缩接口，支持多种图片格式的压缩策略。
"""

from core.compression.manager import CompressionManager
from core.compression.config import CompressionConfig
from core.compression.format import ImageFormat, detect_format
from core.compression.strategy import CompressionStrategy, StaticImageStrategy, GifStrategy

__all__ = [
    "CompressionManager",
    "CompressionConfig",
    "ImageFormat",
    "detect_format",
    "CompressionStrategy",
    "StaticImageStrategy",
    "GifStrategy",
]
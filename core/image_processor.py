# -*- coding: utf-8 -*-
"""
图片处理模块 - 统一压缩管理

使用CompressionManager提供统一的压缩接口，
支持静态图片和GIF动图的压缩。
"""

from typing import Tuple

from astrbot.api import logger

from core.compression import CompressionManager
from core.config import PluginConfig


class ImageProcessor:
    """图片处理器（使用统一压缩管理器）"""

    def __init__(self, config: PluginConfig):
        """
        初始化图片处理器

        Args:
            config: 插件配置
        """
        self.config = config
        self.compression_manager = CompressionManager(config)
        logger.debug("图片处理器初始化完成（使用统一压缩管理器）")

    async def compress_image(self, content: bytes) -> Tuple[bytes, str]:
        """
        压缩图片（统一接口）

        使用CompressionManager自动检测格式并选择合适的压缩策略。

        Args:
            content: 图片内容

        Returns:
            (压缩后的内容, 文件格式)
        """
        return await self.compression_manager.compress(content)

    # 保留旧方法名以兼容性（内部已废弃，建议使用compress_image）
    async def compress_gif(self, content: bytes) -> Tuple[bytes, str]:
        """
        压缩GIF动图（兼容方法，已废弃）

        Args:
            content: GIF图片内容

        Returns:
            (压缩后的内容, 文件格式)
        """
        logger.warning("compress_gif方法已废弃，将自动调用compress_image")
        return await self.compress_image(content)

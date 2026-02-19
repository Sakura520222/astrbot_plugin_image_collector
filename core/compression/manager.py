# -*- coding: utf-8 -*-
"""
压缩管理器模块
"""

from typing import Tuple

from astrbot.api import logger

from core.compression.config import CompressionConfig
from core.compression.format import ImageFormat, detect_format
from core.compression.strategy import CompressionStrategy, GifStrategy, StaticImageStrategy
from core.config import PluginConfig


class CompressionManager:
    """统一压缩管理器"""

    def __init__(self, config: PluginConfig):
        """
        初始化压缩管理器

        Args:
            config: 插件配置对象
        """
        self.config = config
        self.compression_config = CompressionConfig.from_plugin_config(config)

        # 初始化压缩策略
        self._strategies = {
            ImageFormat.GIF: GifStrategy(),
            ImageFormat.JPEG: StaticImageStrategy(),
            ImageFormat.PNG: StaticImageStrategy(),
            ImageFormat.WEBP: StaticImageStrategy(),
            ImageFormat.BMP: StaticImageStrategy(),
        }

        logger.debug("压缩管理器初始化完成")

    async def compress(self, content: bytes) -> Tuple[bytes, str]:
        """
        统一压缩接口

        Args:
            content: 图片内容

        Returns:
            (压缩后的内容, 文件格式)
        """
        # 1. 检测格式
        format_type, is_animated = detect_format(content)
        logger.debug(f"检测到图片格式: {format_type.value}, 是否动图: {is_animated}")

        # 2. 处理未知格式：跳过压缩，返回原始内容
        if format_type == ImageFormat.UNKNOWN:
            logger.warning("图片格式未知，跳过压缩处理")
            return content, "unknown"

        # 3. 选择策略
        strategy = self._select_strategy(format_type, is_animated)

        # 4. 执行压缩
        return await strategy.compress(content, self.compression_config)

    def _select_strategy(
        self, format_type: ImageFormat, is_animated: bool
    ) -> CompressionStrategy:
        """
        根据格式选择压缩策略

        Args:
            format_type: 图片格式
            is_animated: 是否为动图

        Returns:
            压缩策略对象
        """
        # GIF动图特殊处理
        if format_type == ImageFormat.GIF and is_animated:
            if self.compression_config.preserve_gif:
                logger.debug("检测到GIF动图，根据配置跳过压缩")
                # 返回一个不压缩的策略
                return self._create_noop_strategy()
            return self._strategies[ImageFormat.GIF]

        # 其他格式使用静态图片策略
        strategy = self._strategies.get(format_type, self._strategies[ImageFormat.JPEG])
        return strategy

    def _create_noop_strategy(self) -> CompressionStrategy:
        """创建不压缩的策略"""

        class NoopStrategy(CompressionStrategy):
            async def compress(
                self, content: bytes, config: CompressionConfig
            ) -> Tuple[bytes, str]:
                return content, "gif"

        return NoopStrategy()
# -*- coding: utf-8 -*-
"""
压缩配置模块
"""

from dataclasses import dataclass
from typing import Any, Dict

from core.config import PluginConfig


@dataclass
class CompressionConfig:
    """统一压缩配置"""

    # 通用参数
    enable: bool  # 是否启用压缩
    target_size_kb: int  # 目标大小（KB），0表示不限制
    force: bool  # 是否强制压缩所有图片
    preserve_gif: bool  # 是否保留GIF动图不压缩

    # 静态图片参数
    static_max_width: int  # 静态图片最大宽度
    static_max_height: int  # 静态图片最大高度
    jpeg_quality: int  # JPEG质量 (1-100)
    convert_to_jpeg: bool  # 是否转换为JPEG格式
    max_file_size: int  # 最大文件大小（字节）

    # GIF参数（新增独立配置）
    gif_max_width: int  # GIF最大宽度
    gif_max_height: int  # GIF最大高度
    gif_scale_factors: list[float]  # GIF缩放因子列表
    gif_color_levels: list[int]  # GIF颜色深度列表
    gif_min_dimension: int  # GIF最小尺寸限制
    gif_max_frames: int  # GIF最大帧数限制
    gif_frame_skip_method: str  # 抽帧方式：uniform或smart
    gif_convert_to_webp: bool  # 是否转换为WebP格式
    gif_use_mediancut: bool  # 是否使用MEDIANCUT量化算法

    @staticmethod
    def from_plugin_config(config: PluginConfig) -> "CompressionConfig":
        """
        从插件配置创建压缩配置
        
        Args:
            config: 插件配置对象
            
        Returns:
            压缩配置对象
        """
        return CompressionConfig(
            # 通用参数
            enable=config.enable_compression,
            target_size_kb=config.target_size_kb,
            force=config.force_compression,
            preserve_gif=config.preserve_gif,
            # 静态图片参数
            static_max_width=config.max_width,
            static_max_height=config.max_height,
            jpeg_quality=config.jpeg_quality,
            convert_to_jpeg=config.convert_to_jpeg,
            max_file_size=config.max_file_size,
            # GIF参数（使用默认值，后续可通过配置Schema设置）
            gif_max_width=config.get("gif_max_width", 800),
            gif_max_height=config.get("gif_max_height", 600),
            gif_scale_factors=config.get("gif_scale_factors", [1.0, 0.8, 0.6, 0.4]),
            gif_color_levels=config.get("gif_color_levels", [256, 128, 64, 32]),
            gif_min_dimension=config.get("gif_min_dimension", 100),
            gif_max_frames=config.get("gif_max_frames", 0),
            gif_frame_skip_method=config.get("gif_frame_skip_method", "uniform"),
            gif_convert_to_webp=config.get("gif_convert_to_webp", False),
            gif_use_mediancut=config.get("gif_use_mediancut", True),
        )

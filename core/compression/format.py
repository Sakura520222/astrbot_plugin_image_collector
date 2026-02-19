# -*- coding: utf-8 -*-
"""
图片格式枚举和检测工具
"""

from enum import Enum

import io
from PIL import Image

from astrbot.api import logger


class ImageFormat(Enum):
    """图片格式枚举"""

    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    UNKNOWN = "unknown"

    @staticmethod
    def from_pil_format(pil_format: str | None) -> "ImageFormat":
        """
        从PIL格式转换为ImageFormat
        
        Args:
            pil_format: PIL的format属性值
            
        Returns:
            对应的ImageFormat枚举
        """
        if not pil_format:
            return ImageFormat.UNKNOWN
        
        format_map = {
            "JPEG": ImageFormat.JPEG,
            "PNG": ImageFormat.PNG,
            "GIF": ImageFormat.GIF,
            "WEBP": ImageFormat.WEBP,
            "BMP": ImageFormat.BMP,
        }
        return format_map.get(pil_format.upper(), ImageFormat.UNKNOWN)


def detect_format(content: bytes) -> tuple[ImageFormat, bool]:
    """
    检测图片格式和是否为动图
    
    Args:
        content: 图片内容
        
    Returns:
        (格式类型, 是否为动图)
        检测失败时返回 (ImageFormat.UNKNOWN, False)
    """
    try:
        # 使用上下文管理器确保图像对象正确关闭
        with Image.open(io.BytesIO(content)) as img:
            format_type = ImageFormat.from_pil_format(img.format)

            # 检查是否为动图
            is_animated = False
            if format_type == ImageFormat.GIF:
                try:
                    is_animated = getattr(img, "is_animated", False)
                    if not is_animated:
                        # 尝试通过帧数判断
                        try:
                            img.seek(1)
                            img.seek(0)
                            is_animated = True
                        except EOFError:
                            is_animated = False
                except Exception:
                    is_animated = False
            elif format_type == ImageFormat.WEBP:
                # WebP动图检测
                try:
                    is_animated = getattr(img, "is_animated", False)
                except Exception:
                    is_animated = False

            return format_type, is_animated
    except Exception as e:
        # 检测失败时返回 UNKNOWN，避免静默失败
        logger.debug(f"图片格式检测失败: {e}")
        return ImageFormat.UNKNOWN, False

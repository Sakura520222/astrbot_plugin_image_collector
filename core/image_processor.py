# -*- coding: utf-8 -*-
"""
图片处理模块
"""

import io
from typing import Tuple

from PIL import Image

from astrbot.api import logger

from core.config import PluginConfig
from core.exceptions import CompressionError


class ImageProcessor:
    """图片处理器"""

    def __init__(self, config: PluginConfig):
        """
        初始化图片处理器
        
        Args:
            config: 插件配置
        """
        self.config = config

    def detect_image_format(self, content: bytes) -> Tuple[str, bool]:
        """
        检测图片格式和是否为动图
        
        Args:
            content: 图片内容
            
        Returns:
            (格式类型, 是否为动图)
        """
        try:
            img = Image.open(io.BytesIO(content))
            format_type = (img.format or "JPEG").upper()

            # 检查是否为动图
            is_animated = False
            if format_type == "GIF":
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

            return format_type, is_animated
        except Exception:
            return "JPEG", False

    async def compress_gif(
        self,
        content: bytes,
    ) -> Tuple[bytes, str]:
        """
        压缩GIF动图
        
        Args:
            content: GIF图片内容
            
        Returns:
            (压缩后的内容, 文件格式)
        """
        try:
            original_size = len(content)
            img = Image.open(io.BytesIO(content))

            # 获取帧数
            frame_count = 0
            frames = []
            try:
                while True:
                    img.seek(frame_count)
                    frames.append(img.copy())
                    frame_count += 1
            except EOFError:
                pass

            logger.debug(f"GIF动图帧数: {frame_count}")

            # 确定目标大小
            target_size_kb = self.config.target_size_kb
            target_size = target_size_kb * 1024 if target_size_kb > 0 else original_size * 0.8
            logger.debug(f"GIF压缩目标大小: {target_size} 字节")

            # 尺寸压缩（逐步缩小直到满足目标）
            current_frames = frames
            scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

            best_result = content
            best_size = original_size

            for scale in scale_factors:
                if scale < 1.0:
                    # 缩小尺寸
                    new_width = int(img.size[0] * scale)
                    new_height = int(img.size[1] * scale)
                    if new_width < 100 or new_height < 100:
                        break

                    logger.debug(f"尝试缩放: {scale * 100}% ({new_width}x{new_height})")
                    resized_frames = []
                    for frame in current_frames:
                        resized = frame.copy()
                        resized.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                        resized_frames.append(resized)
                    current_frames = resized_frames

                # 尝试不同的颜色深度
                color_levels = [256, 128, 64, 32] if scale <= 0.8 else [256]

                for colors in color_levels:
                    if colors < 256:
                        logger.debug(f"尝试颜色深度: {colors}")
                        quantized_frames = []
                        for frame in current_frames:
                            if frame.mode == "P":
                                reduced = frame.quantize(colors=colors)
                                quantized_frames.append(reduced)
                            else:
                                quantized_frames.append(frame)
                        test_frames = quantized_frames
                    else:
                        test_frames = current_frames

                    # 保存GIF
                    output = io.BytesIO()
                    test_frames[0].save(
                        output,
                        format="GIF",
                        optimize=True,
                        save_all=True,
                        disposal=2,
                        append_images=test_frames[1:] if len(test_frames) > 1 else [],
                        loop=0
                    )
                    compressed = output.getvalue()
                    compressed_size = len(compressed)

                    logger.debug(f"缩放 {scale * 100}%, 颜色 {colors}: {compressed_size} 字节")

                    if compressed_size < best_size:
                        best_result = compressed
                        best_size = compressed_size
                        current_frames = test_frames

                    # 如果达到目标大小，停止
                    if compressed_size <= target_size:
                        logger.info(f"已达到目标大小 {target_size} 字节")
                        break

                if best_size <= target_size:
                    break

            logger.info(f"GIF压缩完成: {original_size} -> {best_size} 字节 ({best_size/1024:.1f} KB)")

            return best_result, "gif"

        except Exception as e:
            logger.error(f"GIF压缩失败: {e}")
            raise CompressionError(f"GIF压缩失败: {e}") from e

    async def compress_image(
        self,
        content: bytes,
    ) -> Tuple[bytes, str]:
        """
        压缩图片（支持强制压缩和目标大小）
        
        Args:
            content: 图片内容
            
        Returns:
            (压缩后的内容, 文件格式)
        """
        try:
            # 检测图片格式和是否为动图
            format_type, is_animated = self.detect_image_format(content)
            original_size = len(content)

            # 如果不是强制压缩，检查文件大小
            if not self.config.force_compression and original_size <= self.config.max_file_size:
                logger.debug(f"图片大小 {original_size} 字节未超过 {self.config.max_file_size} 字节，跳过压缩")
                return content, "jpg" if format_type == "JPEG" else format_type.lower()

            # GIF动图根据配置决定是否压缩
            if format_type == "GIF" and is_animated:
                if self.config.preserve_gif:
                    logger.debug("检测到GIF动图，根据配置跳过压缩")
                    return content, "gif"
                else:
                    logger.debug("检测到GIF动图，将尝试压缩（保持GIF格式）")
                    return await self.compress_gif(content)

            img = Image.open(io.BytesIO(content))

            # 获取原始格式
            original_format = img.format or "JPEG"

            # 尺寸压缩（保持比例）
            if img.size[0] > self.config.max_width or img.size[1] > self.config.max_height:
                img.thumbnail((self.config.max_width, self.config.max_height), Image.Resampling.LANCZOS)

            # 决定输出格式
            output_format = "JPEG" if self.config.convert_to_jpeg else original_format

            # 如果是PNG且要转换为JPEG，需要处理透明通道
            if output_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                # 创建白色背景
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # 确定目标大小
            target_size = self.config.target_size_kb * 1024 if self.config.target_size_kb > 0 else self.config.max_file_size

            # 对于PNG或目标大小很小时，使用更激进的压缩策略
            if output_format == "PNG" or self.config.target_size_kb > 0:
                # 先尝试标准压缩
                output = io.BytesIO()
                img.save(output, format=output_format, optimize=True, **(
                    {"compress_level": 9} if output_format == "PNG" else {"quality": self.config.jpeg_quality}
                ))

                # 如果还不够小，尝试转换为JPEG
                if output.tell() > target_size and output_format == "PNG":
                    logger.debug("PNG压缩后仍过大，尝试转换为JPEG")
                    output_format = "JPEG"

                    # 处理透明通道
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                        img = background
                    elif img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

            # 质量压缩（迭代压缩直到满足文件大小要求）
            output = io.BytesIO()
            current_quality = self.config.jpeg_quality
            min_quality = 5  # 降低最小质量以允许更激进的压缩

            # 如果设置了目标大小，使用更激进的压缩策略
            while current_quality > min_quality:
                output.seek(0)
                output.truncate()

                save_params = {
                    "format": output_format,
                    "quality": current_quality,
                    "optimize": True,
                }

                if output_format == "PNG":
                    save_params.pop("quality")
                    save_params["compress_level"] = 9

                # 在每次迭代中持续降低分辨率
                if output.tell() > target_size:
                    width, height = img.size
                    scale_factor = 0.8 if current_quality > 20 else 0.7  # 质量越低，缩放越激进
                    new_width = max(int(width * scale_factor), 300)  # 最小宽度300
                    new_height = max(int(height * scale_factor), 300)  # 最小高度300
                    if new_width < width or new_height < height:
                        logger.debug(f"降低分辨率: {width}x{height} -> {new_width}x{new_height}")
                        img_copy = img.copy()
                        img_copy.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                        img_copy.save(output, **save_params)
                    else:
                        img.save(output, **save_params)
                else:
                    img.save(output, **save_params)

                compressed_size = output.tell()

                logger.debug(f"压缩质量: {current_quality}, 大小: {compressed_size} 字节, 目标: {target_size} 字节")

                if compressed_size <= target_size:
                    break

                # 根据目标大小调整质量下降步长
                if self.config.target_size_kb > 0:
                    # 如果设置了目标大小，使用更大的步长以更快收敛
                    current_quality -= 20
                else:
                    current_quality -= 10

                # 确保不低于最小质量
                if current_quality < min_quality:
                    current_quality = min_quality

            compressed_size = output.tell()
            logger.info(f"压缩完成: {original_size} -> {compressed_size} 字节 ({compressed_size/1024:.1f} KB)")

            return output.getvalue(), "jpg" if output_format == "JPEG" else "png"

        except Exception as e:
            logger.error(f"压缩图片失败: {e}")
            raise CompressionError(f"压缩图片失败: {e}") from e
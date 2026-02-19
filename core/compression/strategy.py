# -*- coding: utf-8 -*-
"""
压缩策略模块
"""

import asyncio
import io
from abc import ABC, abstractmethod
from typing import Tuple

from PIL import Image

from astrbot.api import logger

from core.compression.config import CompressionConfig
from core.compression.format import ImageFormat
from core.exceptions import CompressionError


class CompressionStrategy(ABC):
    """压缩策略基类"""

    @abstractmethod
    async def compress(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """
        执行压缩

        Args:
            content: 图片内容
            config: 压缩配置

        Returns:
            (压缩后的内容, 文件格式)
        """
        pass

    def _should_compress(
        self, size: int, config: CompressionConfig, is_gif: bool = False
    ) -> bool:
        """
        判断是否需要压缩

        Args:
            size: 文件大小
            config: 压缩配置
            is_gif: 是否为GIF

        Returns:
            是否需要压缩
        """
        if not config.enable:
            return False

        if config.force:
            return True

        # GIF特殊处理：如果配置保留GIF且不是强制压缩，则跳过
        if is_gif and config.preserve_gif:
            return False

        return size > config.max_file_size

    def _resize_image(
        self, img: Image.Image, max_width: int, max_height: int
    ) -> Image.Image:
        """
        调整图片尺寸（保持比例）

        Args:
            img: PIL图片对象
            max_width: 最大宽度
            max_height: 最大高度

        Returns:
            调整后的图片
        """
        if img.size[0] > max_width or img.size[1] > max_height:
            img_copy = img.copy()
            img_copy.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            return img_copy
        return img


class StaticImageStrategy(CompressionStrategy):
    """静态图片压缩策略（JPEG、PNG、WebP等）"""

    async def compress(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """压缩静态图片（异步包装器）"""
        # 将同步的PIL操作放到线程池中执行
        return await asyncio.to_thread(self._compress_sync, content, config)

    def _compress_sync(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """同步压缩静态图片"""
        try:
            original_size = len(content)

            # 检查是否需要压缩
            if not self._should_compress(original_size, config, is_gif=False):
                # 需要检测原始格式以返回正确的扩展名
                img = Image.open(io.BytesIO(content))
                format_type = ImageFormat.from_pil_format(img.format)
                ext_map = {
                    ImageFormat.JPEG: "jpg",
                    ImageFormat.PNG: "png",
                    ImageFormat.WEBP: "webp",
                    ImageFormat.BMP: "bmp",
                }
                ext = ext_map.get(format_type, "jpg")
                logger.debug(
                    f"静态图片大小 {original_size} 字节未超过阈值，跳过压缩"
                )
                return content, ext

            img = Image.open(io.BytesIO(content))
            original_format = img.format or "JPEG"
            format_type = ImageFormat.from_pil_format(original_format)

            # 尺寸压缩
            img = self._resize_image(
                img, config.static_max_width, config.static_max_height
            )

            # 决定输出格式
            output_format = "JPEG" if config.convert_to_jpeg else original_format

            # 处理透明通道（转换为JPEG时需要）
            if output_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img,
                    mask=img.split()[-1] if img.mode == "RGBA" else None,
                )
                img = background
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # 确定目标大小
            target_size = (
                config.target_size_kb * 1024
                if config.target_size_kb > 0
                else config.max_file_size
            )

            # PNG或目标大小很小时的压缩策略
            if output_format == "PNG" or config.target_size_kb > 0:
                output = io.BytesIO()
                img.save(
                    output,
                    format=output_format,
                    optimize=True,
                    **({"compress_level": 9} if output_format == "PNG" else {}),
                )

                # 如果还不够小，尝试转换为JPEG
                if output.tell() > target_size and output_format == "PNG":
                    logger.debug("PNG压缩后仍过大，尝试转换为JPEG")
                    output_format = "JPEG"

                    # 处理透明通道
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(
                            img,
                            mask=img.split()[-1] if img.mode == "RGBA" else None,
                        )
                        img = background
                    elif img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

            # 质量压缩（迭代压缩直到满足文件大小要求）
            output = io.BytesIO()
            current_quality = config.jpeg_quality
            min_quality = 5

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

                # 持续降低分辨率以满足目标大小
                if output.tell() > target_size:
                    width, height = img.size
                    scale_factor = 0.8 if current_quality > 20 else 0.7
                    new_width = max(int(width * scale_factor), 300)
                    new_height = max(int(height * scale_factor), 300)
                    if new_width < width or new_height < height:
                        logger.debug(
                            f"降低分辨率: {width}x{height} -> {new_width}x{new_height}"
                        )
                        img_copy = img.copy()
                        img_copy.thumbnail(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        img_copy.save(output, **save_params)
                    else:
                        img.save(output, **save_params)
                else:
                    img.save(output, **save_params)

                compressed_size = output.tell()

                logger.debug(
                    f"压缩质量: {current_quality}, 大小: {compressed_size} 字节, 目标: {target_size} 字节"
                )

                if compressed_size <= target_size:
                    break

                # 根据目标大小调整质量下降步长
                if config.target_size_kb > 0:
                    current_quality -= 20
                else:
                    current_quality -= 10

                if current_quality < min_quality:
                    current_quality = min_quality

            compressed_size = output.tell()
            logger.info(
                f"静态图片压缩完成: {original_size} -> {compressed_size} 字节 ({compressed_size / 1024:.1f} KB)"
            )

            ext = "jpg" if output_format == "JPEG" else "png"
            return output.getvalue(), ext

        except Exception as e:
            logger.error(f"静态图片压缩失败: {e}")
            raise CompressionError(f"静态图片压缩失败: {e}") from e


class GifStrategy(CompressionStrategy):
    """GIF动图压缩策略"""

    async def compress(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """压缩GIF动图（异步包装器）"""
        # 将同步的PIL操作放到线程池中执行
        return await asyncio.to_thread(self._compress_sync, content, config)

    def _compress_sync(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """同步压缩GIF动图"""
        try:
            original_size = len(content)

            # 检查是否需要压缩
            if not self._should_compress(original_size, config, is_gif=True):
                logger.debug(f"GIF大小 {original_size} 字节未超过阈值，跳过压缩")
                return content, "gif"

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
            target_size = (
                config.target_size_kb * 1024
                if config.target_size_kb > 0
                else original_size * 0.8
            )
            logger.debug(f"GIF压缩目标大小: {target_size} 字节")

            # 尺寸压缩（逐步缩小直到满足目标）
            current_frames = frames
            scale_factors = config.gif_scale_factors

            best_result = content
            best_size = original_size

            for scale in scale_factors:
                if scale < 1.0:
                    # 缩小尺寸
                    new_width = int(img.size[0] * scale)
                    new_height = int(img.size[1] * scale)
                    if (
                        new_width < config.gif_min_dimension
                        or new_height < config.gif_min_dimension
                    ):
                        break

                    logger.debug(f"尝试缩放: {scale * 100}% ({new_width}x{new_height})")
                    resized_frames = []
                    for frame in current_frames:
                        resized = frame.copy()
                        resized.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                        resized_frames.append(resized)
                    current_frames = resized_frames

                # 尝试不同的颜色深度
                color_levels = (
                    config.gif_color_levels if scale <= 0.8 else [256]
                )

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
                        loop=0,
                    )
                    compressed = output.getvalue()
                    compressed_size = len(compressed)

                    logger.debug(
                        f"缩放 {scale * 100}%, 颜色 {colors}: {compressed_size} 字节"
                    )

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

            logger.info(
                f"GIF压缩完成: {original_size} -> {best_size} 字节 ({best_size / 1024:.1f} KB)"
            )

            return best_result, "gif"

        except Exception as e:
            logger.error(f"GIF压缩失败: {e}")
            raise CompressionError(f"GIF压缩失败: {e}") from e
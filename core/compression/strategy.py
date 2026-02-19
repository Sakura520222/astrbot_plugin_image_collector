# -*- coding: utf-8 -*-
"""
压缩策略模块
"""

import asyncio
import io
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

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
    """GIF动图压缩策略（优化版）"""

    async def compress(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """压缩GIF动图（异步包装器）"""
        # 将同步的PIL操作放到线程池中执行
        return await asyncio.to_thread(self._compress_sync, content, config)

    def _compress_sync(
        self, content: bytes, config: CompressionConfig
    ) -> Tuple[bytes, str]:
        """同步压缩GIF动图（优化版）"""
        try:
            original_size = len(content)

            # 检查是否需要压缩
            if not self._should_compress(original_size, config, is_gif=True):
                logger.debug(f"GIF大小 {original_size} 字节未超过阈值，跳过压缩")
                return content, "gif"

            img = Image.open(io.BytesIO(content))

            # 获取帧数和原始尺寸
            frame_count = 0
            frames = []
            try:
                while True:
                    img.seek(frame_count)
                    frames.append(img.copy())
                    frame_count += 1
            except EOFError:
                pass

            original_width, original_height = img.size
            logger.debug(f"GIF动图原始信息: {frame_count}帧, {original_width}x{original_height}px")

            # 确定目标大小
            target_size = (
                config.target_size_kb * 1024
                if config.target_size_kb > 0
                else original_size * 0.8
            )
            logger.debug(f"GIF压缩目标大小: {target_size / 1024:.1f} KB")

            # 步骤1: 帧数优化（抽帧）
            frames = self._optimize_frames(
                frames, frame_count, config.gif_max_frames, config.gif_frame_skip_method
            )
            if len(frames) < frame_count:
                logger.debug(f"帧数优化: {frame_count} -> {len(frames)} 帧")
                frame_count = len(frames)

            # 步骤2: 智能缩放策略（使用二分查找）
            frames, compressed = self._smart_scale_optimization(
                frames, original_width, original_height, target_size, config
            )

            # 如果缩放后达到目标，直接返回
            if compressed and len(compressed) <= target_size:
                logger.info(
                    f"GIF压缩完成(缩放): {original_size} -> {len(compressed)} 字节 ({len(compressed) / 1024:.1f} KB)"
                )
                return compressed, "gif"

            # 步骤3: 颜色深度优化
            frames, compressed = self._color_optimization(
                frames, target_size, config
            )

            # 如果颜色优化后达到目标，直接返回
            if compressed and len(compressed) <= target_size:
                logger.info(
                    f"GIF压缩完成(颜色): {original_size} -> {len(compressed)} 字节 ({len(compressed) / 1024:.1f} KB)"
                )
                return compressed, "gif"

            # 步骤4: 如果启用WebP转换且GIF压缩效果不佳，尝试转换为WebP
            if config.gif_convert_to_webp and len(compressed) > target_size:
                logger.debug("GIF压缩效果不佳，尝试转换为WebP格式")
                webp_result = self._convert_to_webp(frames, target_size, config)
                if webp_result and len(webp_result) <= min(len(compressed), target_size):
                    logger.info(
                        f"GIF转WebP完成: {original_size} -> {len(webp_result)} 字节 ({len(webp_result) / 1024:.1f} KB)"
                    )
                    return webp_result, "webp"

            # 返回最佳结果
            compressed_size = len(compressed)
            logger.info(
                f"GIF压缩完成: {original_size} -> {compressed_size} 字节 ({compressed_size / 1024:.1f} KB)"
            )

            return compressed, "gif"

        except Exception as e:
            logger.error(f"GIF压缩失败: {e}")
            raise CompressionError(f"GIF压缩失败: {e}") from e

    def _optimize_frames(
        self,
        frames: List[Image.Image],
        frame_count: int,
        max_frames: int,
        method: str,
    ) -> List[Image.Image]:
        """
        优化帧数（抽帧）

        Args:
            frames: 帧列表
            frame_count: 总帧数
            max_frames: 最大帧数限制
            method: 抽帧方法（uniform或smart）

        Returns:
            优化后的帧列表
        """
        if max_frames <= 0 or frame_count <= max_frames:
            return frames

        if method == "uniform":
            # 均匀抽帧
            step = frame_count / max_frames
            return [frames[int(i * step)] for i in range(max_frames)]
        else:
            # 智能抽帧：保留首尾帧和关键变化帧
            # 简化实现：保留第一帧、最后一帧，中间均匀采样
            if max_frames >= 2:
                first_frame = frames[0]
                last_frame = frames[-1]
                middle_count = max_frames - 2
                if middle_count > 0:
                    step = (frame_count - 2) / middle_count
                    middle_frames = [frames[int(1 + i * step)] for i in range(middle_count)]
                else:
                    middle_frames = []
                return [first_frame] + middle_frames + [last_frame]
            else:
                return [frames[0]]

    def _smart_scale_optimization(
        self,
        frames: List[Image.Image],
        original_width: int,
        original_height: int,
        target_size: int,
        config: CompressionConfig,
    ) -> Tuple[List[Image.Image], Optional[bytes]]:
        """
        智能缩放优化（使用二分查找）

        Args:
            frames: 帧列表
            original_width: 原始宽度
            original_height: 原始高度
            target_size: 目标大小
            config: 压缩配置

        Returns:
            (优化后的帧列表, 压缩后的内容)
        """
        # 检查是否需要缩放
        if original_width <= config.gif_max_width and original_height <= config.gif_max_height:
            # 不需要缩放，直接保存当前版本
            output = io.BytesIO()
            frames[0].save(
                output,
                format="GIF",
                optimize=True,
                save_all=True,
                disposal=2,
                append_images=frames[1:] if len(frames) > 1 else [],
                loop=0,
            )
            return frames, output.getvalue()

        # 二分查找最佳缩放比例
        low = max(
            config.gif_min_dimension / max(original_width, original_height),
            config.gif_max_height / original_height if original_height > config.gif_max_height else 0,
            config.gif_max_width / original_width if original_width > config.gif_max_width else 0,
        )
        high = 1.0

        best_frames = frames
        best_result = None
        best_size = float("inf")

        # 限制迭代次数避免过长
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations and high - low > 0.05:
            iteration += 1
            scale = (low + high) / 2

            # 缩放帧
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            logger.debug(f"尝试缩放: {scale * 100:.1f}% ({new_width}x{new_height})")

            scaled_frames = []
            for frame in frames:
                resized = frame.copy()
                resized.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                scaled_frames.append(resized)

            # 保存并检查大小
            output = io.BytesIO()
            scaled_frames[0].save(
                output,
                format="GIF",
                optimize=True,
                save_all=True,
                disposal=2,
                append_images=scaled_frames[1:] if len(scaled_frames) > 1 else [],
                loop=0,
            )
            compressed = output.getvalue()
            compressed_size = len(compressed)

            logger.debug(f"缩放结果: {compressed_size / 1024:.1f} KB")

            if compressed_size < best_size:
                best_frames = scaled_frames
                best_result = compressed
                best_size = compressed_size

            # 调整搜索范围
            if compressed_size > target_size:
                high = scale
            else:
                low = scale

            # 早期终止：如果已经达到目标
            if compressed_size <= target_size:
                break

        return best_frames, best_result

    def _color_optimization(
        self,
        frames: List[Image.Image],
        target_size: int,
        config: CompressionConfig,
    ) -> Tuple[List[Image.Image], bytes]:
        """
        颜色深度优化

        Args:
            frames: 帧列表
            target_size: 目标大小
            config: 压缩配置

        Returns:
            (优化后的帧列表, 压缩后的内容)
        """
        best_frames = frames
        best_result = None
        best_size = float("inf")

        # 先保存当前版本作为基准
        output = io.BytesIO()
        frames[0].save(
            output,
            format="GIF",
            optimize=True,
            save_all=True,
            disposal=2,
            append_images=frames[1:] if len(frames) > 1 else [],
            loop=0,
        )
        best_result = output.getvalue()
        best_size = len(best_result)

        if best_size <= target_size:
            return frames, best_result

        # 尝试不同的颜色深度
        for colors in config.gif_color_levels:
            if colors >= 256:
                continue

            logger.debug(f"尝试颜色深度: {colors}")

            quantized_frames = []
            for frame in frames:
                if frame.mode == "P":
                    # 使用更优的量化算法
                    if config.gif_use_mediancut:
                        try:
                            # 尝试使用MEDIANCUT量化方法
                            reduced = frame.quantize(
                                colors=colors,
                                method=Image.Quantize.MEDIANCUT if hasattr(Image.Quantize, 'MEDIANCUT') else None,
                            )
                            quantized_frames.append(reduced)
                        except Exception as e:
                            logger.debug(f"MEDIANCUT量化失败，使用默认方法: {e}")
                            reduced = frame.quantize(colors=colors)
                            quantized_frames.append(reduced)
                    else:
                        reduced = frame.quantize(colors=colors)
                        quantized_frames.append(reduced)
                else:
                    quantized_frames.append(frame)

            # 保存并检查
            output = io.BytesIO()
            quantized_frames[0].save(
                output,
                format="GIF",
                optimize=True,
                save_all=True,
                disposal=2,
                append_images=quantized_frames[1:] if len(quantized_frames) > 1 else [],
                loop=0,
            )
            compressed = output.getvalue()
            compressed_size = len(compressed)

            logger.debug(f"颜色深度 {colors}: {compressed_size / 1024:.1f} KB")

            if compressed_size < best_size:
                best_frames = quantized_frames
                best_result = compressed
                best_size = compressed_size

            # 早期终止
            if compressed_size <= target_size:
                break

        return best_frames, best_result

    def _convert_to_webp(
        self,
        frames: List[Image.Image],
        target_size: int,
        config: CompressionConfig,
    ) -> Optional[bytes]:
        """
        转换为WebP格式

        Args:
            frames: 帧列表
            target_size: 目标大小
            config: 压缩配置

        Returns:
            WebP格式的内容，失败返回None
        """
        try:
            output = io.BytesIO()

            # WebP支持更好的压缩
            save_params = {
                "format": "WebP",
                "save_all": True,
                "duration": 100,  # 默认帧持续时间
                "loop": 0,
                "quality": 85,
                "method": 6,  # 最高压缩级别
            }

            frames[0].save(
                output,
                **save_params,
                append_images=frames[1:] if len(frames) > 1 else [],
            )

            webp_content = output.getvalue()
            logger.debug(f"WebP转换成功: {len(webp_content) / 1024:.1f} KB")

            return webp_content

        except Exception as e:
            logger.debug(f"WebP转换失败（可能Pillow版本不支持）: {e}")
            return None
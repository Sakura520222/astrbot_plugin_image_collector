# -*- coding: utf-8 -*-
"""
文件操作模块
"""

import time
from pathlib import Path
from typing import List, Optional

from astrbot.api import logger

from core.config import PluginConfig
from core.dedup_service import DedupService
from core.exceptions import FileOperationError
from core.hash_calculator import HashCalculator
from core.image_processor import ImageProcessor
from core.index_manager import IndexManager
from core.path_manager import PathManager


class FileHandler:
    """文件处理器"""

    def __init__(
        self,
        config: PluginConfig,
        path_manager: PathManager,
        index_manager: IndexManager,
        hash_calculator: HashCalculator,
        image_processor: ImageProcessor,
        dedup_service: DedupService,
    ):
        """
        初始化文件处理器
        
        Args:
            config: 插件配置
            path_manager: 路径管理器
            index_manager: 索引管理器
            hash_calculator: 哈希计算器
            image_processor: 图片处理器
            dedup_service: 去重服务
        """
        self.config = config
        self.path_manager = path_manager
        self.index_manager = index_manager
        self.hash_calculator = hash_calculator
        self.image_processor = image_processor
        self.dedup_service = dedup_service

    async def save_image(
        self,
        content: bytes,
        filename: str,
        user_dir: Path,
    ) -> Optional[Path]:
        """
        保存图片到文件
        
        Args:
            content: 图片内容
            filename: 文件名
            user_dir: 用户目录
            
        Returns:
            保存的文件路径，失败返回None
        """
        try:
            filepath = user_dir / filename
            with open(filepath, "wb") as f:
                f.write(content)
            logger.info(f"图片已保存: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            raise FileOperationError(f"保存图片失败: {e}") from e

    def get_user_images(self, user_dir: Path) -> List[str]:
        """
        获取用户的所有图片文件
        
        Args:
            user_dir: 用户图片目录
            
        Returns:
            图片文件路径列表
        """
        if not user_dir.exists():
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        images = [
            str(f) for f in user_dir.iterdir() if f.suffix.lower() in image_extensions
        ]
        logger.debug(f"用户 {user_dir.name} 有 {len(images)} 张图片")
        return images

    async def process_image(
        self,
        content: bytes,
        platform: str,
        group_id: str,
        user_id: str,
    ) -> Optional[Path]:
        """
        处理图片：去重、压缩、保存
        
        Args:
            content: 图片内容
            platform: 平台名称
            group_id: 群组ID
            user_id: 用户ID
            
        Returns:
            保存的文件路径，如果图片重复则返回None
        """
        # 1. 先计算哈希（使用原始内容进行去重检测）
        md5_hash = self.hash_calculator.calculate_md5(content)
        phash_data = {}
        if self.config.enable_phash:
            try:
                phash_data = self.hash_calculator.calculate_perceptual_hash(content)
            except Exception as e:
                logger.warning(f"计算感知哈希失败，将跳过感知哈希去重: {e}")

        # 2. 获取用户目录
        user_dir = self.path_manager.get_user_dir(platform, group_id, user_id)
        dedup_scope = self.config.dedup_scope

        # 3. 检查去重（如果已存在，直接返回，避免重复压缩）
        is_dup = await self.dedup_service.is_duplicate(
            md5_hash, phash_data, user_dir, dedup_scope
        )
        if is_dup:
            logger.info(f"图片已存在，跳过压缩和保存 (MD5: {md5_hash})")
            return None

        # 4. 自动压缩（只有在图片不重复时才压缩）
        if self.config.enable_compression:
            try:
                content, file_format = await self.image_processor.compress_image(content)
            except Exception as e:
                logger.error(f"压缩图片失败，保存原始图片: {e}")
                file_format = "jpg"
        else:
            file_format = "jpg"

        # 5. 保存图片
        filename = f"{int(time.time())}_{md5_hash}.{file_format}"
        filepath = await self.save_image(content, filename, user_dir)

        if filepath:
            # 6. 更新索引
            index_path = self.path_manager.get_index_path(user_dir, dedup_scope)
            await self.index_manager.update_index(
                index_path,
                md5_hash,
                phash_data,
                filename,
                self.config.enable_phash,
            )

        return filepath
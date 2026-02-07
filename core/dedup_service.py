# -*- coding: utf-8 -*-
"""
去重服务模块
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import imagehash

from astrbot.api import logger

from core.config import PluginConfig
from core.exceptions import DuplicateImageError
from core.hash_calculator import HashCalculator
from core.index_manager import IndexManager
from core.path_manager import PathManager

if TYPE_CHECKING:
    from core.config import PluginConfig


class DedupService:
    """去重服务"""

    def __init__(
        self,
        config: PluginConfig,
        path_manager: PathManager,
        index_manager: IndexManager,
        hash_calculator: HashCalculator,
    ):
        """
        初始化去重服务
        
        Args:
            config: 插件配置
            path_manager: 路径管理器
            index_manager: 索引管理器
            hash_calculator: 哈希计算器
        """
        self.config = config
        self.path_manager = path_manager
        self.index_manager = index_manager
        self.hash_calculator = hash_calculator

    async def is_duplicate(
        self,
        md5_hash: str,
        phash_data: Dict[str, str],
        user_dir: Path,
        dedup_scope: str,
    ) -> bool:
        """
        检查图片是否重复
        
        Args:
            md5_hash: MD5哈希值
            phash_data: 感知哈希数据
            user_dir: 用户目录
            dedup_scope: 去重范围
            
        Returns:
            是否重复
        """
        # 如果两种去重都禁用，直接返回不重复
        if not self.config.enable_md5_dedup and not self.config.enable_phash:
            return False

        index_path = self.path_manager.get_index_path(user_dir, dedup_scope)
        index = self.index_manager.load_index(index_path)

        # MD5精确去重
        if self.config.enable_md5_dedup:
            if md5_hash in index.get("md5_index", {}):
                logger.debug(f"图片MD5已存在: {md5_hash}")
                return True

        # 感知哈希相似去重
        if self.config.enable_phash and phash_data:
            threshold = self.config.phash_threshold
            phash_index = index.get("phash_index", {})

            for existing_phash, existing_md5 in phash_index.items():
                try:
                    h1 = imagehash.hex_to_hash(existing_phash)
                    h2 = imagehash.hex_to_hash(phash_data.get("phash", ""))
                    distance = h1 - h2
                    if distance <= threshold:
                        logger.debug(f"图片感知哈希相似，距离: {distance}, 阈值: {threshold}")
                        return True
                except Exception as e:
                    logger.warning(f"比较感知哈希失败: {e}")
                    continue

        return False
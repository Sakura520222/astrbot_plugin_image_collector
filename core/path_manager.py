# -*- coding: utf-8 -*-
"""
路径管理模块
"""

from pathlib import Path

from astrbot.api import logger

from core.config import PluginConfig


class PathManager:
    """路径管理器"""

    def __init__(self, config: PluginConfig):
        """
        初始化路径管理器
        
        Args:
            config: 插件配置
        """
        self.config = config
        self._data_dir = config.get_data_dir()

    def get_user_dir(self, platform: str, group_id: str, user_id: str) -> Path:
        """
        获取用户图片存储目录
        
        Args:
            platform: 平台名称
            group_id: 群组ID
            user_id: 用户ID
            
        Returns:
            用户图片目录路径
        """
        user_dir = self._data_dir / platform / group_id / user_id / "images"
        user_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"用户目录: {user_dir}")
        return user_dir

    def get_index_path(self, user_dir: Path, scope: str) -> Path:
        """
        获取索引文件路径
        
        Args:
            user_dir: 用户目录
            scope: 索引范围 (user/group/global)
            
        Returns:
            索引文件路径
        """
        if scope == "user":
            return user_dir.parent / "index.json"
        elif scope == "group":
            return user_dir.parent.parent / "group_index.json"
        else:  # global
            return self._data_dir / "global_index.json"
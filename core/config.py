# -*- coding: utf-8 -*-
"""
配置管理模块
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger


class PluginConfig:
    """插件配置包装类，提供统一的配置访问接口"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        
        Args:
            config: AstrBot传入的配置字典
        """
        self.config = config if config is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self.config.get(key, default)

    # 图片保存配置
    @property
    def enable_save(self) -> bool:
        """是否启用图片保存"""
        return self.get("enable_save", True)

    @property
    def enable_random_send(self) -> bool:
        """是否启用随机发送"""
        return self.get("enable_random_send", True)

    @property
    def save_on_image_msg(self) -> bool:
        """是否在发送图片消息时也触发随机发送"""
        return self.get("save_on_image_msg", False)

    # 去重配置
    @property
    def dedup_scope(self) -> str:
        """去重范围: user(用户), group(群组), global(全局)"""
        return self.get("dedup_scope", "user")

    @property
    def enable_md5_dedup(self) -> bool:
        """是否启用MD5精确去重"""
        return self.get("enable_md5_dedup", True)

    @property
    def enable_phash(self) -> bool:
        """是否启用感知哈希去重"""
        return self.get("enable_phash", True)

    @property
    def phash_threshold(self) -> int:
        """感知哈相似阈值（汉明距离）"""
        return self.get("phash_threshold", 5)

    # 压缩配置
    @property
    def enable_compression(self) -> bool:
        """是否启用压缩"""
        return self.get("enable_compression", True)

    @property
    def max_width(self) -> int:
        """最大宽度"""
        return self.get("max_width", 1920)

    @property
    def max_height(self) -> int:
        """最大高度"""
        return self.get("max_height", 1080)

    @property
    def jpeg_quality(self) -> int:
        """JPEG质量 (1-100)"""
        return self.get("jpeg_quality", 85)

    @property
    def max_file_size(self) -> int:
        """最大文件大小（字节）"""
        return self.get("max_file_size", 2) * 1024 * 1024

    @property
    def convert_to_jpeg(self) -> bool:
        """是否转换为JPEG格式"""
        return self.get("convert_to_jpeg", True)

    @property
    def force_compression(self) -> bool:
        """是否强制压缩所有图片"""
        return self.get("force_compression", False)

    @property
    def target_size_kb(self) -> int:
        """目标大小（KB），0表示不限制"""
        return self.get("target_size_kb", 0)

    @property
    def preserve_gif(self) -> bool:
        """是否保留GIF动图不压缩"""
        return self.get("preserve_gif", True)

    # 随机发送配置
    @property
    def random_send_probability(self) -> float:
        """随机发送概率 (0.0-1.0)"""
        return self.get("random_send_probability", 0.3)

    @property
    def min_images(self) -> int:
        """最少图片数量才触发随机发送"""
        return self.get("min_images", 1)

    # 消息过滤配置
    @property
    def exclude_words(self) -> List[str]:
        """排除词列表"""
        return self.get("exclude_words", [])

    @property
    def ignore_commands(self) -> bool:
        """是否忽略指令消息"""
        return self.get("ignore_commands", True)

    @property
    def command_prefixes(self) -> List[str]:
        """指令前缀列表"""
        return self.get("command_prefixes", ["/"])

    def set_data_dir(self, data_dir: Path) -> None:
        """
        设置数据目录
        
        Args:
            data_dir: 数据目录路径
        """
        self._data_dir = data_dir
        logger.info(f"图片收集插件数据目录: {data_dir}")

    def get_data_dir(self) -> Path:
        """
        获取数据目录
        
        Returns:
            数据目录路径
        """
        if not hasattr(self, "_data_dir"):
            raise RuntimeError("数据目录未初始化，请先调用 set_data_dir()")
        return self._data_dir
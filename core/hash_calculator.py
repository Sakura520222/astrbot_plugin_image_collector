# -*- coding: utf-8 -*-
"""
哈希计算模块
"""

import hashlib
import io
from typing import Dict

import imagehash
from PIL import Image

from astrbot.api import logger

from core.exceptions import HashCalculationError


class HashCalculator:
    """哈希计算器"""

    def calculate_md5(self, content: bytes) -> str:
        """
        计算MD5哈希
        
        Args:
            content: 文件内容
            
        Returns:
            MD5哈希值（十六进制字符串）
        """
        try:
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"计算MD5失败: {e}")
            raise HashCalculationError(f"计算MD5失败: {e}") from e

    def calculate_perceptual_hash(self, content: bytes) -> Dict[str, str]:
        """
        计算感知哈希
        
        Args:
            content: 图片内容
            
        Returns:
            包含多种感知哈希的字典
        """
        try:
            with Image.open(io.BytesIO(content)) as img:
                return {
                    "ahash": str(imagehash.average_hash(img)),
                    "phash": str(imagehash.phash(img)),
                    "dhash": str(imagehash.dhash(img)),
                    "whash": str(imagehash.whash(img)),
                }
        except Exception as e:
            logger.error(f"计算感知哈希失败: {e}")
            raise HashCalculationError(f"计算感知哈希失败: {e}") from e

    def is_similar_image(
        self, phash1: Dict[str, str], phash2: Dict[str, str], threshold: int
    ) -> bool:
        """
        判断两张图片是否相似
        
        Args:
            phash1: 第一张图片的感知哈希
            phash2: 第二张图片的感知哈希
            threshold: 相似阈值（汉明距离）
            
        Returns:
            是否相似
        """
        if not phash1 or not phash2:
            return False

        try:
            h1 = imagehash.hex_to_hash(phash1.get("phash", ""))
            h2 = imagehash.hex_to_hash(phash2.get("phash", ""))
            distance = h1 - h2
            is_similar = distance <= threshold
            if is_similar:
                logger.debug(f"图片相似，汉明距离: {distance}, 阈值: {threshold}")
            return is_similar
        except Exception as e:
            logger.error(f"判断图片相似性失败: {e}")
            return False
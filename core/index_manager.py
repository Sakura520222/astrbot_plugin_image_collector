# -*- coding: utf-8 -*-
"""
索引管理模块
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

from astrbot.api import logger

from core.exceptions import FileOperationError


class IndexManager:
    """索引管理器"""

    def load_index(self, index_path: Path) -> Dict:
        """
        加载索引文件
        
        Args:
            index_path: 索引文件路径
            
        Returns:
            索引数据字典
        """
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                    logger.debug(f"加载索引文件: {index_path}")
                    return index
            except json.JSONDecodeError as e:
                logger.error(f"索引文件JSON格式错误: {e}")
                return {"md5_index": {}, "phash_index": {}}
            except Exception as e:
                logger.error(f"加载索引文件失败: {e}")
                raise FileOperationError(f"加载索引文件失败: {e}") from e
        return {"md5_index": {}, "phash_index": {}}

    async def save_index(self, index_path: Path, index: Dict) -> None:
        """
        保存索引文件
        
        Args:
            index_path: 索引文件路径
            index: 索引数据字典
        """
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            logger.debug(f"保存索引文件: {index_path}")
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}")
            raise FileOperationError(f"保存索引文件失败: {e}") from e

    async def update_index(
        self,
        index_path: Path,
        md5_hash: str,
        phash_data: Dict[str, str],
        filename: str,
        enable_phash: bool,
    ) -> None:
        """
        更新索引文件
        
        Args:
            index_path: 索引文件路径
            md5_hash: MD5哈希值
            phash_data: 感知哈希数据
            filename: 文件名
            enable_phash: 是否启用感知哈希
        """
        index = self.load_index(index_path)

        # 更新MD5索引
        if "md5_index" not in index:
            index["md5_index"] = {}
        index["md5_index"][md5_hash] = {
            "filename": filename,
            "timestamp": int(time.time()),
        }

        # 更新感知哈希索引
        if enable_phash and phash_data:
            if "phash_index" not in index:
                index["phash_index"] = {}
            index["phash_index"][phash_data.get("phash", "")] = md5_hash

        await self.save_index(index_path, index)
        logger.debug(f"更新索引: {md5_hash} -> {filename}")
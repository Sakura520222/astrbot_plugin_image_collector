# -*- coding: utf-8 -*-
"""AstrBot 图片收集插件 - 自动收集群友图片并随机回复"""

import asyncio
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import imagehash
from PIL import Image

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Image as CompImage
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.api.star import Context, Star, register


@register(
    "image_collector",
    "Cline",
    "自动收集群友发送的图片，支持去重和压缩，并在用户发送消息时随机回复其图库中的图片",
    "1.0.0",
    "https://github.com/your-repo/astrbot_plugin_image_collector",
)
class ImageCollectorPlugin(Star):
    """图片收集插件"""

    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        
        # 初始化数据目录
        astrbot_data_path = get_astrbot_data_path()
        plugin_name = "image_collector"  # 插件名称
        data_dir_str = os.path.join(astrbot_data_path, "plugin_data", plugin_name)
        self.data_dir = Path(data_dir_str)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"图片收集插件已加载，数据目录: {self.data_dir}")

    def get_user_dir(self, platform: str, group_id: str, user_id: str) -> Path:
        """获取用户图片存储目录"""
        user_dir = (
            self.data_dir / platform / group_id / user_id / "images"
        )
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_index_path(self, user_dir: Path, scope: str) -> Path:
        """获取索引文件路径"""
        if scope == "user":
            return user_dir.parent / "index.json"
        elif scope == "group":
            return user_dir.parent.parent / "group_index.json"
        else:  # global
            return self.data_dir / "global_index.json"

    def load_index(self, index_path: Path) -> Dict:
        """加载索引文件"""
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载索引文件失败: {e}")
                return {"md5_index": {}, "phash_index": {}}
        return {"md5_index": {}, "phash_index": {}}

    async def save_index(self, index_path: Path, index: Dict):
        """保存索引文件"""
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}")

    def calculate_md5(self, content: bytes) -> str:
        """计算MD5哈希"""
        return hashlib.md5(content).hexdigest()

    def calculate_perceptual_hash(self, content: bytes) -> Dict[str, str]:
        """计算感知哈希"""
        try:
            img = Image.open(io.BytesIO(content))
            return {
                "ahash": str(imagehash.average_hash(img)),
                "phash": str(imagehash.phash(img)),
                "dhash": str(imagehash.dhash(img)),
                "whash": str(imagehash.whash(img)),
            }
        except Exception as e:
            logger.error(f"计算感知哈希失败: {e}")
            return {}

    def is_similar_image(
        self, phash1: Dict[str, str], phash2: Dict[str, str], threshold: int
    ) -> bool:
        """判断两张图片是否相似"""
        if not phash1 or not phash2:
            return False
        
        try:
            h1 = imagehash.hex_to_hash(phash1.get("phash", ""))
            h2 = imagehash.hex_to_hash(phash2.get("phash", ""))
            distance = h1 - h2
            return distance <= threshold
        except Exception as e:
            logger.error(f"判断图片相似性失败: {e}")
            return False

    def detect_image_format(self, content: bytes) -> tuple[str, bool]:
        """检测图片格式和是否为动图"""
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

    async def compress_image(
        self,
        content: bytes,
        max_width: int,
        max_height: int,
        quality: int,
        max_file_size: int,
        convert_to_jpeg: bool,
    ) -> tuple[bytes, str]:
        """压缩图片（GIF动图可配置是否跳过压缩）"""
        try:
            # 检测图片格式和是否为动图
            format_type, is_animated = self.detect_image_format(content)
            
            # GIF动图根据配置决定是否压缩
            if format_type == "GIF" and is_animated:
                if self.config.get("preserve_gif", True):
                    logger.debug("检测到GIF动图，根据配置跳过压缩")
                    return content, "gif"
                else:
                    logger.debug("检测到GIF动图，但配置允许压缩，继续处理")
            
            img = Image.open(io.BytesIO(content))
            
            # 获取原始格式
            original_format = img.format or "JPEG"
            
            # 尺寸压缩（保持比例）
            if img.size[0] > max_width or img.size[1] > max_height:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # 决定输出格式
            output_format = "JPEG" if convert_to_jpeg else original_format
            
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
            
            # 质量压缩（迭代压缩直到满足文件大小要求）
            output = io.BytesIO()
            current_quality = quality
            
            while current_quality > 10:
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
                
                img.save(output, **save_params)
                
                if output.tell() <= max_file_size:
                    break
                
                current_quality -= 10
            
            return output.getvalue(), "jpg" if output_format == "JPEG" else "png"
            
        except Exception as e:
            logger.error(f"压缩图片失败: {e}")
            return content, "jpg"

    async def download_image(self, url: str) -> Optional[bytes]:
        """下载图片"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.read()
        except Exception as e:
            logger.error(f"下载图片失败 {url}: {e}")
        return None

    async def is_duplicate(
        self,
        md5_hash: str,
        phash_data: Dict[str, str],
        user_dir: Path,
        dedup_scope: str,
    ) -> bool:
        """检查图片是否重复"""
        if not self.config.get("enable_md5_dedup", True) and not self.config.get(
            "enable_phash", True
        ):
            return False
        
        index_path = self.get_index_path(user_dir, dedup_scope)
        index = self.load_index(index_path)
        
        # MD5精确去重
        if self.config.get("enable_md5_dedup", True):
            if md5_hash in index.get("md5_index", {}):
                logger.debug(f"图片MD5已存在: {md5_hash}")
                return True
        
        # 感知哈希相似去重
        if self.config.get("enable_phash", True) and phash_data:
            threshold = self.config.get("phash_threshold", 5)
            phash_index = index.get("phash_index", {})
            
            for existing_phash, existing_md5 in phash_index.items():
                try:
                    h1 = imagehash.hex_to_hash(existing_phash)
                    h2 = imagehash.hex_to_hash(phash_data.get("phash", ""))
                    distance = h1 - h2
                    if distance <= threshold:
                        logger.debug(f"图片感知哈希相似，距离: {distance}")
                        return True
                except Exception:
                    continue
        
        return False

    async def update_index(
        self,
        user_dir: Path,
        md5_hash: str,
        phash_data: Dict[str, str],
        filename: str,
        dedup_scope: str,
    ):
        """更新索引文件"""
        index_path = self.get_index_path(user_dir, dedup_scope)
        index = self.load_index(index_path)
        
        # 更新MD5索引
        if "md5_index" not in index:
            index["md5_index"] = {}
        index["md5_index"][md5_hash] = {
            "filename": filename,
            "timestamp": int(time.time()),
        }
        
        # 更新感知哈希索引
        if self.config.get("enable_phash", True) and phash_data:
            if "phash_index" not in index:
                index["phash_index"] = {}
            index["phash_index"][phash_data.get("phash", "")] = md5_hash
        
        await self.save_index(index_path, index)

    async def save_image(
        self,
        content: bytes,
        filename: str,
        user_dir: Path,
    ) -> Optional[Path]:
        """保存图片到文件"""
        try:
            filepath = user_dir / filename
            with open(filepath, "wb") as f:
                f.write(content)
            return filepath
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            return None

    def get_user_images(self, user_dir: Path) -> List[str]:
        """获取用户的所有图片文件"""
        if not user_dir.exists():
            return []
        
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        images = [
            str(f) for f in user_dir.iterdir() if f.suffix.lower() in image_extensions
        ]
        return images

    async def process_image(
        self,
        content: bytes,
        platform: str,
        group_id: str,
        user_id: str,
    ) -> Optional[Path]:
        """处理图片：压缩、去重、保存"""
        # 1. 自动压缩
        if self.config.get("enable_compression", True):
            content, file_format = await self.compress_image(
                content,
                max_width=self.config.get("max_width", 1920),
                max_height=self.config.get("max_height", 1080),
                quality=self.config.get("jpeg_quality", 85),
                max_file_size=self.config.get("max_file_size", 2) * 1024 * 1024,
                convert_to_jpeg=self.config.get("convert_to_jpeg", True),
            )
        else:
            file_format = "jpg"
        
        # 2. 计算哈希
        md5_hash = self.calculate_md5(content)
        phash_data = {}
        if self.config.get("enable_phash", True):
            phash_data = self.calculate_perceptual_hash(content)
        
        # 3. 获取用户目录
        user_dir = self.get_user_dir(platform, group_id, user_id)
        dedup_scope = self.config.get("dedup_scope", "user")
        
        # 4. 检查去重
        if await self.is_duplicate(md5_hash, phash_data, user_dir, dedup_scope):
            logger.info(f"图片已存在，跳过保存 (MD5: {md5_hash})")
            return None
        
        # 5. 保存图片
        filename = f"{int(time.time())}_{md5_hash}.{file_format}"
        filepath = await self.save_image(content, filename, user_dir)
        
        if filepath:
            logger.info(f"图片已保存: {filepath}")
            # 6. 更新索引
            await self.update_index(user_dir, md5_hash, phash_data, filename, dedup_scope)
        
        return filepath

    def should_exclude_message(self, message_str: str) -> bool:
        """检查消息是否应该被排除"""
        exclude_words = self.config.get("exclude_words", [])
        for word in exclude_words:
            if word in message_str:
                return True
        return False

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息"""
        # 检查是否启用功能
        if not self.config.get("enable_save", True) and not self.config.get(
            "enable_random_send", True
        ):
            return
        
        # 获取基本信息
        platform = event.get_platform_name()
        group_id = event.message_obj.group_id or "default"
        user_id = event.get_sender_id()
        message_str = event.message_str or ""
        
        # 检查是否应该排除
        if self.should_exclude_message(message_str):
            return
        
        # 解析消息链中的图片
        message_chain = event.message_obj.message
        images = [msg for msg in message_chain if isinstance(msg, CompImage)]
        
        # 处理图片
        saved_count = 0
        if images and self.config.get("enable_save", True):
            for img in images:
                content = None
                
                # 从URL下载
                if img.url:
                    content = await self.download_image(img.url)
                # 从本地文件读取
                elif hasattr(img, "file") and img.file:
                    try:
                        with open(img.file, "rb") as f:
                            content = f.read()
                    except Exception as e:
                        logger.error(f"读取本地图片失败: {e}")
                
                if content:
                    await self.process_image(content, platform, group_id, user_id)
                    saved_count += 1
        
        # 随机发送逻辑
        if self.config.get("enable_random_send", True):
            should_send = (
                (not images and saved_count == 0) or self.config.get("save_on_image_msg", False)
            )
            
            if should_send and not self.should_exclude_message(message_str):
                user_dir = self.get_user_dir(platform, group_id, user_id)
                user_images = self.get_user_images(user_dir)
                min_images = self.config.get("min_images", 1)
                
                if len(user_images) >= min_images:
                    random_image = random.choice(user_images)
                    try:
                        yield event.image_result(random_image)
                        logger.info(f"已随机发送图片: {random_image}")
                    except Exception as e:
                        logger.error(f"发送图片失败: {e}")

    @filter.command("图片统计")
    async def image_stats(self, event: AstrMessageEvent):
        """查看当前群收集的图片统计"""
        platform = event.get_platform_name()
        group_id = event.message_obj.group_id or "default"
        group_dir = self.data_dir / platform / group_id
        
        if not group_dir.exists():
            yield event.plain_result("当前群还没有收集任何图片")
            return
        
        stats = []
        total_images = 0
        total_users = 0
        
        for user_dir in group_dir.iterdir():
            if user_dir.is_dir():
                images_dir = user_dir / "images"
                if images_dir.exists():
                    images = self.get_user_images(images_dir)
                    count = len(images)
                    if count > 0:
                        total_users += 1
                        total_images += count
                        stats.append(f"  {user_dir.name}: {count}张")
        
        if total_images == 0:
            yield event.plain_result("当前群还没有收集任何图片")
        else:
            response = f"当前群图片统计：\n总用户数: {total_users}\n总图片数: {total_images}\n\n按用户统计:\n" + "\n".join(stats)
            yield event.plain_result(response)

    @filter.command("我的图片")
    async def my_images(self, event: AstrMessageEvent):
        """查看我收集的图片数量"""
        platform = event.get_platform_name()
        group_id = event.message_obj.group_id or "default"
        user_id = event.get_sender_id()
        
        user_dir = self.get_user_dir(platform, group_id, user_id)
        user_images = self.get_user_images(user_dir)
        
        yield event.plain_result(f"你已收集 {len(user_images)} 张图片")

    async def terminate(self):
        """插件卸载时调用"""
        logger.info("图片收集插件已卸载")
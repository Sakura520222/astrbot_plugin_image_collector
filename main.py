"""AstrBot 图片收集插件 - 自动收集群友图片并随机回复"""

import os
import sys

# 添加插件目录到 sys.path 以支持模块导入
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

import random

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Image as CompImage
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.api.star import Context, Star, register

from core.config import PluginConfig
from core.dedup_service import DedupService
from core.file_handler import FileHandler
from core.hash_calculator import HashCalculator
from core.image_downloader import ImageDownloader
from core.image_processor import ImageProcessor
from core.index_manager import IndexManager
from core.message_filter import MessageFilter
from core.path_manager import PathManager


@register(
    "image_collector",
    "Sakura520222",
    "自动收集群友发送的图片，支持去重和压缩，并在用户发送消息时随机回复其图库中的图片",
    "1.2.0",
    "https://github.com/Sakura520222/astrbot_plugin_image_collector",
)
class ImageCollectorPlugin(Star):
    """图片收集插件"""

    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        
        # 初始化配置管理器
        self.config = PluginConfig(config if config is not None else {})
        
        # 初始化数据目录
        from pathlib import Path
        astrbot_data_path_str = get_astrbot_data_path()
        plugin_name = "image_collector"
        data_dir_str = os.path.join(astrbot_data_path_str, "plugin_data", plugin_name)
        data_dir = Path(data_dir_str)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.config.set_data_dir(data_dir)
        
        # 初始化各个管理器
        self.path_manager = PathManager(self.config)
        self.index_manager = IndexManager()
        self.hash_calculator = HashCalculator()
        self.image_processor = ImageProcessor(self.config)
        self.image_downloader = ImageDownloader()
        self.dedup_service = DedupService(
            self.config,
            self.path_manager,
            self.index_manager,
            self.hash_calculator,
        )
        self.file_handler = FileHandler(
            self.config,
            self.path_manager,
            self.index_manager,
            self.hash_calculator,
            self.image_processor,
            self.dedup_service,
        )
        self.message_filter = MessageFilter(self.config)
        
        logger.info(f"图片收集插件已加载，数据目录: {data_dir}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息"""
        # 检查是否启用功能
        if not self.config.enable_save and not self.config.enable_random_send:
            return
        
        # 获取基本信息
        platform = event.get_platform_name()
        group_id = event.message_obj.group_id or "default"
        user_id = event.get_sender_id()
        message_str = event.message_str or ""
        
        # 检查是否应该排除
        if self.message_filter.should_exclude_message(message_str):
            return
        
        # 解析消息链中的图片
        message_chain = event.message_obj.message
        images = [msg for msg in message_chain if isinstance(msg, CompImage)]
        
        # 处理图片
        saved_count = 0
        if images and self.config.enable_save:
            for img in images:
                content = None
                
                # 从URL下载
                if img.url:
                    content = await self.image_downloader.download_image(img.url)
                # 从本地文件读取
                elif hasattr(img, "file") and img.file:
                    try:
                        with open(img.file, "rb") as f:
                            content = f.read()
                    except Exception as e:
                        logger.error(f"读取本地图片失败: {e}")
                
                if content:
                    await self.file_handler.process_image(
                        content, platform, group_id, user_id
                    )
                    saved_count += 1
        
        # 随机发送逻辑
        if self.config.enable_random_send:
            should_send = (
                (not images and saved_count == 0) or self.config.save_on_image_msg
            )
            
            if should_send and not self.message_filter.should_exclude_message(message_str):
                # 检查概率
                probability = self.config.random_send_probability
                if random.random() < probability:
                    user_dir = self.path_manager.get_user_dir(
                        platform, group_id, user_id
                    )
                    user_images = self.file_handler.get_user_images(user_dir)
                    
                    if len(user_images) >= self.config.min_images:
                        random_image = random.choice(user_images)
                        try:
                            yield event.image_result(random_image)
                            logger.info(
                                f"已随机发送图片: {random_image} (概率触发)"
                            )
                        except Exception as e:
                            logger.error(f"发送图片失败: {e}")
                else:
                    logger.debug(f"随机发送未触发 (概率: {probability})")

    @filter.command("图片统计")
    async def image_stats(self, event: AstrMessageEvent):
        """查看当前群收集的图片统计"""
        platform = event.get_platform_name()
        group_id = event.message_obj.group_id or "default"
        data_dir = self.config.get_data_dir()
        group_dir = data_dir / platform / group_id
        
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
                    images = self.file_handler.get_user_images(images_dir)
                    count = len(images)
                    if count > 0:
                        total_users += 1
                        total_images += count
                        stats.append(f"  {user_dir.name}: {count}张")
        
        if total_images == 0:
            yield event.plain_result("当前群还没有收集任何图片")
        else:
            response = (
                f"当前群图片统计：\n总用户数: {total_users}\n总图片数: {total_images}\n\n按用户统计:\n"
                + "\n".join(stats)
            )
            yield event.plain_result(response)

    @filter.command("我的图片")
    async def my_images(self, event: AstrMessageEvent):
        """查看我收集的图片数量"""
        platform = event.get_platform_name()
        group_id = event.message_obj.group_id or "default"
        user_id = event.get_sender_id()
        
        user_dir = self.path_manager.get_user_dir(platform, group_id, user_id)
        user_images = self.file_handler.get_user_images(user_dir)
        
        yield event.plain_result(f"你已收集 {len(user_images)} 张图片")

    async def terminate(self):
        """插件卸载时调用"""
        logger.info("图片收集插件已卸载")
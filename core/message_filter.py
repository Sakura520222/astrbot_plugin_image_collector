# -*- coding: utf-8 -*-
"""
消息过滤模块
"""

from astrbot.api import logger

from core.config import PluginConfig


class MessageFilter:
    """消息过滤器"""

    def __init__(self, config: PluginConfig):
        """
        初始化消息过滤器
        
        Args:
            config: 插件配置
        """
        self.config = config

    def should_exclude_message(self, message_str: str) -> bool:
        """
        检查消息是否应该被排除
        
        Args:
            message_str: 消息文本
            
        Returns:
            是否应该排除
        """
        if not message_str:
            return False

        # 检查排除词
        exclude_words = self.config.exclude_words
        for word in exclude_words:
            if word in message_str:
                logger.debug(f"消息包含排除词 '{word}'，跳过处理")
                return True

        # 检查是否为指令（以配置的前缀开头）
        # 注意：由于AstrBot框架的特性，指令消息的message_str可能不包含前缀
        # 此检测主要用于手动输入包含前缀的消息
        if self.config.ignore_commands:
            prefixes = self.config.command_prefixes
            trimmed_msg = message_str.strip()

            for prefix in prefixes:
                if trimmed_msg.startswith(prefix):
                    logger.debug(f"消息以指令前缀 '{prefix}' 开头，跳过处理")
                    return True

        return False
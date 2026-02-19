# -*- coding: utf-8 -*-
"""
图片下载模块
"""

from typing import Optional

import aiohttp

from astrbot.api import logger

from core.exceptions import DownloadError


class ImageDownloader:
    """图片下载器"""

    def __init__(self):
        """初始化图片下载器"""
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        获取或创建共享的 HTTP 会话
        
        Returns:
            aiohttp.ClientSession 实例
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """关闭 HTTP 会话，释放资源"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def download_image(self, url: str) -> Optional[bytes]:
        """
        下载图片
        
        Args:
            url: 图片URL
            
        Returns:
            图片内容，失败返回None
        """
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    logger.debug(f"成功下载图片: {url}, 大小: {len(content)} 字节")
                    return content
                else:
                    logger.warning(f"下载图片失败，状态码: {resp.status}, URL: {url}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"网络请求失败 {url}: {e}")
            raise DownloadError(f"下载图片失败: {e}") from e
        except Exception as e:
            logger.error(f"下载图片失败 {url}: {e}")
            raise DownloadError(f"下载图片失败: {e}") from e
# -*- coding: utf-8 -*-
"""
图片收集插件异常定义
"""


class ImageCollectorError(Exception):
    """图片收集插件基础异常类"""
    pass


class DownloadError(ImageCollectorError):
    """图片下载异常"""
    pass


class ProcessingError(ImageCollectorError):
    """图片处理异常"""
    pass


class HashCalculationError(ImageCollectorError):
    """哈希计算异常"""
    pass


class CompressionError(ImageCollectorError):
    """图片压缩异常"""
    pass


class FileOperationError(ImageCollectorError):
    """文件操作异常"""
    pass


class DuplicateImageError(ImageCollectorError):
    """重复图片异常"""
    pass
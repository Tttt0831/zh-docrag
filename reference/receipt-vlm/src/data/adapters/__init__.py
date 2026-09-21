"""
数据集适配器模块
将 CORD 数据集映射到统一的 6 字段 schema
"""

from .cord_adapter import CORDAdapter

__all__ = ['CORDAdapter']

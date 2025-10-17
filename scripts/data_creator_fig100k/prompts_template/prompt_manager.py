"""
Fig100k Prompt模板管理器
用于加载和渲染Jinja2模板
"""
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any


class Fig100kPromptManager:
    """Fig100k Prompt模板管理器"""
    
    def __init__(self, template_dir: str = None):
        """
        初始化Prompt管理器
        
        Args:
            template_dir: 模板目录路径，默认为当前目录
        """
        if template_dir is None:
            template_dir = Path(__file__).parent
        
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render_prompt(self, template_name: str, **kwargs) -> str:
        """
        渲染prompt模板
        
        Args:
            template_name: 模板文件名（不包含.j2扩展名）
            **kwargs: 模板变量
            
        Returns:
            渲染后的prompt字符串
        """
        try:
            template = self.env.get_template(f"{template_name}.j2")
            return template.render(**kwargs)
        except Exception as e:
            raise Exception(f"Failed to render template {template_name}: {e}")
    
    def get_thinking_combined_prompt(self, caption: str, context: str) -> str:
        """获取合并的thinking分析prompt（同时生成short和long）"""
        return self.render_prompt(
            "thinking_combined",
            caption=caption,
            context=context
        )


# 全局prompt管理器实例
fig100k_prompt_manager = Fig100kPromptManager()

"""
Prompt模板管理器
用于加载和渲染Jinja2模板
"""
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any


class PromptManager:
    """Prompt模板管理器"""
    
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
    
    def get_figure_classification_prompt(self, paper_title: str, figures_text: str) -> str:
        """获取图片分类prompt"""
        return self.render_prompt(
            "figure_classification",
            paper_title=paper_title,
            figures_text=figures_text
        )
    
    def get_diagram_analysis_prompt(self, caption: str, context: str) -> str:
        """获取diagram分析prompt"""
        return self.render_prompt(
            "diagram_analysis",
            caption=caption,
            context=context
        )
    
    def get_diagram_description_prompt(self, caption: str, context: str) -> str:
        """获取diagram描述prompt"""
        return self.render_prompt(
            "diagram_description",
            caption=caption,
            context=context
        )
    
    def get_thinking_generation_prompt(self, caption: str, context: str, visual_analysis: str) -> str:
        """获取thinking生成prompt"""
        return self.render_prompt(
            "thinking_generation",
            caption=caption,
            context=context,
            visual_analysis=visual_analysis
        )


# 全局prompt管理器实例
prompt_manager = PromptManager()

#!/usr/bin/env python3
"""
安装SentenceTransformer依赖的脚本
用于避免Azure API rate限制问题
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        print(f"📦 正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        return False

def test_sentence_transformer():
    """测试SentenceTransformer是否正常工作"""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        print("🧪 测试SentenceTransformer...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📱 检测到设备: {device}")
        
        # 测试加载模型
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        print("✅ 模型加载成功")
        
        # 测试编码
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        print(f"✅ 编码测试成功，embedding维度: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ SentenceTransformer测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 SentenceTransformer 安装脚本")
    print("=" * 50)
    print("目标: 避免Azure API rate限制，使用本地embedding模型")
    print()
    
    # 检查是否已安装
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        print("✅ SentenceTransformer 已安装")
        
        if test_sentence_transformer():
            print("\n🎉 SentenceTransformer 已就绪！")
            print("现在可以运行 hybrid_diagram_reasoner.py 了")
            return
    except ImportError:
        print("❌ SentenceTransformer 未安装，开始安装...")
    
    # 安装依赖
    packages = [
        "sentence-transformers",
        "torch",
        "torchvision",  # 可选，但推荐
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    if success_count == len(packages):
        print("🎉 所有依赖安装完成！")
        
        # 测试安装结果
        if test_sentence_transformer():
            print("\n✅ 安装验证成功！")
            print("现在可以运行 hybrid_diagram_reasoner.py 了")
        else:
            print("\n❌ 安装验证失败，请检查错误信息")
    else:
        print(f"❌ 安装失败，只有 {success_count}/{len(packages)} 个包安装成功")

if __name__ == "__main__":
    main()

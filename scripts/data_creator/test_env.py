#!/usr/bin/env python3
"""
测试环境变量加载
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def test_env_loading():
    """测试环境变量加载"""
    print("🔍 测试环境变量加载")
    print("=" * 50)
    
    # 尝试从多个位置加载 .env 文件
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / "CoTT" / ".env",
        Path("/Users/suny0a/Proj/MM-Reasoning/CoTT/.env"),
        Path("/home/t2vg-a100-G2-0/yasheng/CoTT/.env"),
    ]
    
    print("🔍 检查 .env 文件路径:")
    for i, env_path in enumerate(env_paths, 1):
        exists = env_path.exists()
        print(f"   {i}. {env_path} - {'✅ 存在' if exists else '❌ 不存在'}")
        if exists:
            print(f"      📄 文件大小: {env_path.stat().st_size} bytes")
    
    print("\n🔍 尝试加载环境变量:")
    loaded = False
    for env_path in env_paths:
        if env_path.exists():
            print(f"📄 加载: {env_path}")
            load_dotenv(env_path)
            loaded = True
            break
    
    if not loaded:
        print("❌ 未找到任何 .env 文件")
        return
    
    print("\n🔍 检查环境变量:")
    env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "PAPYRUS_ENDPOINT",
        "PAPYRUS_VERIFY_SCOPE",
        "PAPYRUS_CLIENT_ID"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if "KEY" in var or "TOKEN" in var:
                print(f"   ✅ {var}: {value[:20]}...")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: 未设置")

if __name__ == "__main__":
    test_env_loading()

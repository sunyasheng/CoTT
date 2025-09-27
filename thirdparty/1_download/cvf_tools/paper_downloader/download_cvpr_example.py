#!/usr/bin/env python3
"""
CVF论文下载示例脚本
支持下载CVPR、ICCV、WACV、ACCV等会议的论文
"""

import os
import sys

# 添加lib目录到Python路径
root_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

from code.paper_downloader_CVF import download_paper

def download_cvpr_papers():
    """下载CVPR 2024论文示例"""
    print("开始下载CVPR 2024论文...")
    
    download_paper(
        year=2024,
        conference='CVPR',
        save_dir='/Users/suny0a/Proj/CoTT/papers/CVPR',
        is_download_main_paper=True,      # 下载主论文
        is_download_supplement=True,      # 下载补充材料
        time_step_in_seconds=5,          # 下载间隔5秒
        is_download_main_conference=True, # 下载主会议论文
        is_download_workshops=False,      # 暂时不下载workshops
        downloader=None,                  # 使用Python requests
    )
    print("CVPR 2024论文下载完成！")

def download_iccv_papers():
    """下载ICCV 2023论文示例"""
    print("开始下载ICCV 2023论文...")
    
    download_paper(
        year=2023,
        conference='ICCV',
        save_dir='/Users/suny0a/Proj/CoTT/papers/ICCV',
        is_download_main_paper=True,
        is_download_supplement=True,
        time_step_in_seconds=5,
        is_download_main_conference=True,
        is_download_workshops=False,
        downloader=None,
    )
    print("ICCV 2023论文下载完成！")

def download_wacv_papers():
    """下载WACV 2024论文示例"""
    print("开始下载WACV 2024论文...")
    
    download_paper(
        year=2024,
        conference='WACV',
        save_dir='/Users/suny0a/Proj/CoTT/papers/WACV',
        is_download_main_paper=True,
        is_download_supplement=True,
        time_step_in_seconds=5,
        is_download_main_conference=True,
        is_download_workshops=False,
        downloader=None,
    )
    print("WACV 2024论文下载完成！")

if __name__ == '__main__':
    print("CVF论文下载器示例")
    print("=" * 50)
    
    # 创建保存目录
    base_dir = '/Users/suny0a/Proj/CoTT/papers'
    os.makedirs(base_dir, exist_ok=True)
    
    # 选择要下载的会议
    print("请选择要下载的会议：")
    print("1. CVPR 2024")
    print("2. ICCV 2023") 
    print("3. WACV 2024")
    print("4. 全部下载")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == '1':
        download_cvpr_papers()
    elif choice == '2':
        download_iccv_papers()
    elif choice == '3':
        download_wacv_papers()
    elif choice == '4':
        download_cvpr_papers()
        download_iccv_papers()
        download_wacv_papers()
    else:
        print("无效选择，退出程序")

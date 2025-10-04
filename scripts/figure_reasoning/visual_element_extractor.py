import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import traceback

def extract_pdf_images_bbox_smart(pdf_path: str) -> list:
    """
    智能提取PDF中所有图片的bbox信息。
    专门处理复杂PDF，如vis.pdf这种有大量重复图片实例的情况。
    """
    doc = fitz.open(pdf_path)
    elements_info = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"📄 处理第 {page_num + 1} 页")
        
        page_rect = page.rect
        print(f"   页面尺寸: {page_rect.width} x {page_rect.height}")
        
        # 方法1: 使用display list获取实际绘制的图片，按z-order去重
        print("   🔍 使用display list方法分析实际绘制的图片（按z-order去重）...")
        try:
            rawdict = page.get_text("rawdict")
            displaylist_images = []
            
            for i, block in enumerate(rawdict.get("blocks", [])):
                if block.get('type') == 1:  # 1 表示 image
                    bbox = fitz.Rect(block["bbox"])
                    xref = block.get("xref", f"unknown_displaylist_{i}")
                    
                    # 只保留在页面范围内的图片
                    if (bbox.x0 >= 0 and bbox.y0 >= 0 and 
                        bbox.x1 <= page_rect.width and bbox.y1 <= page_rect.height):
                        
                        displaylist_images.append({
                            'page': page_num + 1,
                            'type': 'image',
                            'method': 'displaylist',
                            'xref': xref,
                            'block_index': i,  # 记录在display list中的顺序
                            'bbox': {k: getattr(bbox, k) for k in ['x0', 'y0', 'x1', 'y1', 'width', 'height']},
                            'page_size': {'width': page_rect.width, 'height': page_rect.height}
                        })
                        print(f"   📄 Display list - 图片 xref={xref}, bbox=({bbox.x0:.1f}, {bbox.y0:.1f}, {bbox.x1:.1f}, {bbox.y1:.1f}), block_index={i}")
            
            # 精确的可见性分析：只保留最终显示中真正可见的图片
            print(f"   🔄 精确可见性分析，原始图片数: {len(displaylist_images)}")
            
            # 按block_index排序（绘制顺序）
            sorted_images = sorted(displaylist_images, key=lambda x: x['block_index'])
            visible_images = []
            
            for i, current_img in enumerate(sorted_images):
                current_bbox = current_img['bbox']
                is_visible = True
                
                # 检查是否被后续绘制的图片完全遮挡
                for j in range(i + 1, len(sorted_images)):
                    later_img = sorted_images[j]
                    later_bbox = later_img['bbox']
                    
                    # 检查later_img是否完全覆盖current_img
                    if (later_bbox['x0'] <= current_bbox['x0'] and 
                        later_bbox['y0'] <= current_bbox['y0'] and
                        later_bbox['x1'] >= current_bbox['x1'] and 
                        later_bbox['y1'] >= current_bbox['y1']):
                        # 被完全遮挡
                        is_visible = False
                        break
                
                if is_visible:
                    visible_images.append(current_img)
            
            elements_info.extend(visible_images)
            print(f"   ✅ 精确可见性分析后保留 {len(visible_images)} 个真正可见的图片")
            
        except Exception as e:
            print(f"   ⚠️ Display list方法失败: {e}")
        
        # 只保留display list方法找到的实际图片，跳过其他方法
        print("   ✅ 仅使用display list方法，跳过其他分析...")
    
    doc.close()
    
    # 最终去重：基于bbox位置去重
    final_elements = []
    seen_positions = set()
    
    for element_info in elements_info:
        bbox = element_info['bbox']
        # 使用精确到小数点后1位的去重标准
        position_key = (round(bbox['x0'], 1), round(bbox['y0'], 1), round(bbox['x1'], 1), round(bbox['y1'], 1))
        
        if position_key not in seen_positions:
            seen_positions.add(position_key)
            final_elements.append(element_info)
    
    print(f"   🔄 最终去重后保留 {len(final_elements)} 个唯一图片/区域")
    return final_elements

def visualize_bbox_on_png(png_path: str, bbox_info: list, output_path: str = None):
    """在PNG图片上可视化bbox位置"""
    try:
        img = Image.open(png_path).convert("RGB")
        img_width, img_height = img.size
        
        fig, ax = plt.subplots(1, figsize=(12, 12 * img_height / img_width))
        ax.imshow(img)
        ax.set_title("Smart PDF Images BBox Visualization")
        ax.axis('off')

        # 颜色列表，用于区分不同的bbox
        colors = plt.cm.get_cmap('tab20', len(bbox_info))

        for i, info in enumerate(bbox_info):
            pdf_bbox = info['bbox']
            page_size = info['page_size']
            method = info.get('method', 'unknown')
            note = info.get('note', '')
            
            # 将PDF坐标转换为PNG图片坐标
            scale_x = img_width / page_size['width']
            scale_y = img_height / page_size['height']
            
            x0_png = pdf_bbox['x0'] * scale_x
            y0_png = pdf_bbox['y0'] * scale_y
            x1_png = pdf_bbox['x1'] * scale_x
            y1_png = pdf_bbox['y1'] * scale_y
            
            width_png = x1_png - x0_png
            height_png = y1_png - y0_png
            
            rect = plt.Rectangle((x0_png, y0_png), width_png, height_png,
                                linewidth=2, edgecolor=colors(i), facecolor='none',
                                label=f"{info.get('type', 'unknown')} {i+1} ({method})")
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x0_png, y0_png - 5, f"{info.get('type', 'unknown')} {i+1}",
                    color=colors(i), fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            
            print(f"   📦 区域 {i+1} ({info.get('type', 'unknown')}, {method}): PDF bbox=({pdf_bbox['x0']:.1f}, {pdf_bbox['y0']:.1f}, {pdf_bbox['x1']:.1f}, {pdf_bbox['y1']:.1f}) {note}")

        # 创建图例
        if len(bbox_info) > 0:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            for handle, label in zip(handles, labels):
                unique_labels[label] = handle
            
            display_handles = list(unique_labels.values())[:20]
            display_labels = list(unique_labels.keys())[:20]

            ax.legend(display_handles, display_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 可视化结果已保存到: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    except FileNotFoundError:
        print(f"❌ 错误: PNG文件未找到: {png_path}")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        print(f"   详细错误: {traceback.format_exc()}")

def analyze_pdf_and_visualize(pdf_path: str, png_path: str, output_dir: str = "smart_bbox_analysis"):
    """
    智能分析PDF并可视化bbox。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("============================================================")
    print(f"🔍 智能分析PDF: {pdf_path}")
    print(f"📸 对应PNG: {png_path}")
    print("============================================================")

    bbox_info = extract_pdf_images_bbox_smart(pdf_path)
    
    json_output_path = os.path.join(output_dir, os.path.basename(pdf_path).replace(".pdf", "_smart_bbox_info.json"))
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_info, f, ensure_ascii=False, indent=2)
    print(f"💾 BBox信息已保存到: {json_output_path}")

    if bbox_info:
        visualization_output_path = os.path.join(output_dir, os.path.basename(pdf_path).replace(".pdf", "_smart_bbox_visualization.png"))
        visualize_bbox_on_png(png_path, bbox_info, visualization_output_path)
        print(f"🖼️ 可视化结果已保存到: {visualization_output_path}")
    else:
        print("⚠️ 未找到任何图片或可识别区域，跳过可视化。")

    print("\n✅ 智能分析完成!")
    print(f"📊 找到 {len(bbox_info)} 个图片/区域")
    if bbox_info:
        print(f"🖼️ 可视化文件: {visualization_output_path}")
    print(f"📄 详细信息: {json_output_path}")

    return bbox_info

if __name__ == "__main__":
    # 测试vis.pdf
    pdf_file = "/Users/suny0a/Proj/MM-Reasoning/CoTT/workspace/papers_latex/arXiv-2509.11171v1/samples/vis.pdf"
    png_file = "/Users/suny0a/Proj/MM-Reasoning/CoTT/workspace/papers_latex/arXiv-2509.11171v1/samples/vis.png"
    analyze_pdf_and_visualize(pdf_file, png_file)

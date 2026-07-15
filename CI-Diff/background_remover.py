"""
背景去除脚本
此脚本用于去除图像背景，只保留马的主体部分

使用方法:
1. 安装依赖: pip install rembg pillow onnxruntime
2. 运行脚本: python horse_background_remover.py
"""

import os
from rembg import remove
from PIL import Image
import io


def remove_background(input_path, output_path):
    """
    去除单张图片背景
    
    Args:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
    """
    try:
        # 打开输入图片
        with open(input_path, 'rb') as img_file:
            input_image = img_file.read()
        
        # 去除背景
        output_image_data = remove(input_image)
        
        # 保存处理后的图片
        img = Image.open(io.BytesIO(output_image_data))
        
        # 确保输出路径的目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存图片
        img.save(output_path, format='PNG')
        
        print(f"背景去除完成: {output_path}")
    except Exception as e:
        print(f"处理图片时出错: {e}")


def process_images(input_folder, output_folder):
    """
    批量处理文件夹中的所有图片
    
    Args:
        input_folder (str): 输入图片文件夹路径
        output_folder (str): 输出处理后图片的文件夹路径
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        # 确保文件是图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f"no_bg_{filename.split('.')[0]}.png")
            
            remove_background(input_image_path, output_image_path)


def main():
    """
    主函数 - 处理testimg文件夹中的图片
    """
    # 处理单张图片
    input_path = "testimg/f1.jpg"
    output_path = "output_examples/single_result/f1_no_bg.png"
    
    if os.path.exists(input_path):
        remove_background(input_path, output_path)
    else:
        print(f"输入图片不存在: {input_path}")
    
    # 批量处理
    input_folder = "testimg"
    output_folder = "output_examples/batch_results"
    
    if os.path.exists(input_folder):
        process_images(input_folder, output_folder)
        print(f"批量处理完成，结果保存在: {output_folder}")
    else:
        print(f"输入文件夹不存在: {input_folder}")


if __name__ == "__main__":
    main()
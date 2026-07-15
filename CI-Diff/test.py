import torch
import os
import gc
from PIL import Image
# 基础模型导入
from modelscope import StableDiffusion3Pipeline
# 自定义模型导入
from models.transformer_sd3 import SD3Transformer2DModel
from infer_sd35_large_ipa import StableDiffusion3Pipeline as CustomStableDiffusion3Pipeline
from background_remover import remove_background, process_images
from diffusers.utils import load_image
import cv2
import numpy as np
def generate_edge_map(image_path, low_threshold=100, high_threshold=200, output_path=None):
    """
    Generate edge detection map of the image (based on Canny algorithm)
    
    Args:
        image_path (str): Path to the input image
        low_threshold (int): Low threshold for Canny edge detection, controls detection sensitivity
        high_threshold (int): High threshold for Canny edge detection, controls edge connectivity
        output_path (str, optional): Save path for the edge map; if None, it will not be saved
    
    Returns:
        PIL.Image: Generated edge detection map (RGB format)
    """
    try:
        image = load_image(image_path)
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    image_np = np.array(image)
    
    # Check if the image is in RGB format; convert it if not
    if len(image_np.shape) == 2:  
        gray_image = image_np
    else:  
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    
    # Convert to PIL image
    edge_image = Image.fromarray(edges_rgb)
    
    # save image
    if output_path:
        try:
            edge_image.save(output_path)
            print(f"Edge map saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save edge map: {e}")
    
    return edge_image


def generate_images(p2_list, p3_list, base_result_dir, base_seed=42):
    """
    Universal function for batch image generation
    :param p2_list: Subject list (base image prompts)
    :param p3_list: rare list (target image prompts)
    :param base_result_dir: Base save directory (e.g., "single_4action")
    :param base_seed: Random seed to ensure reproducible results
    """
    # -------------------------- If no reference image is provided, use the following code to generate --------------------------
    """
    common_dir = os.path.join("ref_img", "Base_common")
    background_dir = os.path.join("ref_img", "background")
    edge_dir = os.path.join("ref_img", "edge")
    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    # Load base model
    pipe_base = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # p2_list
    for idx, prompt in enumerate(p2_list):
        generator = torch.Generator("cuda").manual_seed(base_seed)
        image = pipe_base(
            prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
            generator=generator
        ).images[0]
        save_path = os.path.join(common_dir, f"{idx}.jpg")
        image.save(save_path)
        
    for i in range(len(p2_list)):  # Assume that p2 and p3 have the same length.
        # Load reference image (corresponding image in the common directory)
        ref_img_path = os.path.join(common_dir, f"{i}.jpg")
        ref_img = Image.open(ref_img_path).convert('RGB')
        output_path = f"{background_dir}/{i}.jpg"
        remove_background(ref_img_path, output_path)
        print(f"Single image processing completed, results saved in: {output_path}")
        input_image_path = output_path  # Replace with your input image path
        output_edge_path = f"{edge_dir}/{i}.jpg"  # Edge map output path
        edge_map = generate_edge_map(
            image_path=input_image_path,
            low_threshold=80,
            high_threshold=200,
            output_path=output_edge_path
        )
        ref_img = Image.open(output_edge_path).convert('RGB')

    del pipe_base
    gc.collect()
    torch.cuda.empty_cache()
    """
    # -------------------------- Generate target images (our) --------------------------
    our_dir = os.path.join(base_result_dir, "our")
    os.makedirs(our_dir, exist_ok=True)
    # load edge image
    print(f"  Generate target images.（CI-Diff）...")
    # Model path setup (adjust according to the actual path)
    model_path = 'XXXXX/stabilityai/stable-diffusion-3.5-large'
    ip_adapter_path = 'XXXXXX/IP-Adapter7/ip-adapter.bin'
    image_encoder_path = "XXXXXX/google/siglip-so400m-patch14-384"

    # Load custom model and IP-Adapter
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe_custom = CustomStableDiffusion3Pipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe_custom.init_ipadapter(
        ip_adapter_path=ip_adapter_path, 
        image_encoder_path=image_encoder_path, 
        nb_token=64, 
    )

    # Batch generate images in the our directory
    for i in range(len(p2_list)):
        # Get prompt
        prompt = p3_list[i]
        prompt_subject = p2_list[i]
        ref_img_path = f"ref_img/edge/{i}.jpg" 
        ref_img = Image.open(ref_img_path).convert('RGB')
        
        # Generate image
        image = pipe_custom(
            width=1024,
            height=1024,
            prompt=prompt,
            prompt_subject=prompt_subject,
            clip_image=ref_img,
            guidance_scale_rare=5.0,
            guidance_scale_subject=4.0,
            num_inference_steps=30,
            generator=torch.Generator("cuda").manual_seed(base_seed),
            ipadapter_scale=0.2,
        ).images[0]
        
        # Save the image.
        save_path = os.path.join(our_dir, f"{i}.jpg")
        image.save(save_path)
        # print(f"  The image has been saved：{save_path}")

    # Clear the custom model GPU memory.
    del pipe_custom, transformer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{base_result_dir} Processing completed！\n")


# -------------------------- Define the dataset to be processed. --------------------------
# single_1property (Original Data)
p3_list_single_1property = [
    "A hairy octopus",          
]

p2_list_single_1property = [
    "An octopus",                            
]
if __name__ == "__main__":
    tasks = [
        (p2_list_single_1property, p3_list_single_1property, "single_1property"),
    ]

    for p2, p3, dir_name in tasks:
        generate_images(p2, p3, dir_name)

from PIL import Image
def combine_images_horizontally(image_list, output_path):
    """
    将图像列表中的所有图像水平拼接成一张图片。

    :param image_list: 包含Pillow图像对象的列表。
    :param output_path: 输出文件的路径。
    """
    # 图像列表为空时的处理
    if not image_list:
        raise ValueError("The list of images is empty.")

    # 获取第一张图片的尺寸作为参考
    first_image = image_list[0]
    width, height = first_image.size

    # 计算新图像的总宽度和高度
    total_width = sum(image.width for image in image_list)
    new_height = height  # 所有图像的高度假设相同

    # 创建一个新的空白图像，宽度为所有图像宽度之和，高度为单张图像的高度
    combined_image = Image.new('RGB', (total_width, new_height), (255, 255, 255))

    # 初始化x坐标
    x_offset = 0

    # 遍历图像列表，并将每个图像粘贴到新图像上
    for image in image_list:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # 保存拼接后的图像
    combined_image.save(output_path) 

def read_promt(filename,num):

    import json
    from PIL import Image
    import os

    # 假设你的JSON文件名为data.json
    #filename = '/mnt/workspace/qinshiyang/PixArt-alpha/test/200data/partition/200data_info.json'

    # 初始化一个空列表来存储prompt字段
    prompts = []

    images = []

    # 打开JSON文件并读取数据
    with open(filename, 'r', encoding='utf-8') as file:
        # 将整个文件的内容加载为JSON对象
        data = json.load(file)
        
        # 假设JSON文件中的数据是一个列表
        # 读取前100条数据的prompt字段
        for item in data[:num]:
            # 确保item确实包含prompt字段
            if 'prompt' in item:
                prompts.append(item['prompt'])
            if 'path' in item:
                images.append(Image.open(item['path']))
    return images,prompts

filename = '/mnt/workspace/qinshiyang/PixArt-alpha/test/200data/partition/200data_info.json'
origin_images,prompt=read_promt(filename,100)

def merge_images_into_sheets(original_images, generated_images1, generated_images2, output_dir, rows=3, cols=10, resize_size=(512, 512)):
    from PIL import Image
    import os
    """
    将三个图像列表中的图像调整为指定大小后，每30张合并为一版，原版图像在第一行，生成图像1在第二行，生成图像2在第三行，每行有10列。

    :param original_images: 原版图像的列表，每个元素是PIL.Image对象。
    :param generated_images1: 生成图像1的列表，每个元素是PIL.Image对象。
    :param generated_images2: 生成图像2的列表，每个元素是PIL.Image对象。
    :param output_dir: 输出文件夹路径，合并后的图像将保存在此文件夹中。
    :param rows: 每版图像中的行数。
    :param cols: 每版图像中的列数。
    :param resize_size: 调整图像大小到指定的宽度和高度。
    :return: 无返回值，合并后的图像将保存在指定的输出文件夹中。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 确保所有列表长度相同或为3的倍数
    if len(original_images) % cols != 0 or len(generated_images1) % cols != 0 or len(generated_images2) % cols != 0:
        raise ValueError("所有图像列表的长度必须能被列数整除。")

    # 计算总的版数
    total_sheets = (len(original_images) + cols - 1) // cols

    for sheet_index in range(total_sheets):
        # 创建合并后的图像
        combined_image = Image.new('RGB', (cols * resize_size[0], rows * resize_size[1]), color='white')

        # 合并原版图像
        for i in range(cols):
            image_index = sheet_index * cols + i
            if image_index < len(original_images):
                img = original_images[image_index]
                img_resized = img.resize(resize_size)
                combined_image.paste(img_resized, (i * resize_size[0], 0 * resize_size[1]))

        # 合并生成图像1
        for i in range(cols):
            image_index = sheet_index * cols + i
            if image_index < len(generated_images1):
                img = generated_images1[image_index]
                img_resized = img.resize(resize_size)
                combined_image.paste(img_resized, (i * resize_size[0], 1 * resize_size[1]))

        # 合并生成图像2
        for i in range(cols):
            image_index = sheet_index * cols + i
            if image_index < len(generated_images2):
                img = generated_images2[image_index]
                img_resized = img.resize(resize_size)
                combined_image.paste(img_resized, (i * resize_size[0], 2 * resize_size[1]))

        # 生成唯一的文件名
        output_filename = f"compare_{sheet_index + 1}.jpg"
        output_filepath = os.path.join(output_dir, output_filename)

        # 保存合并后的图像
        combined_image.save(output_filepath)
        print(f"合并后的图像已保存为: {output_filepath}")

import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL,Transformer2DModel

#微调
# If loading a LoRA model
transformer = Transformer2DModel.from_pretrained("/mnt/workspace/qinshiyang/PixArt-alpha/test/diffuser/fine_transformer_6000", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "Your-LoRA-Model-Path")
finetune_pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
# del transformer

# Enable memory optimizations.
finetune_pipe.enable_model_cpu_offload()

# Enable memory optimizations.
finetune_pipe.enable_model_cpu_offload()

finetune_gen_images = finetune_pipe(prompt).images


#预训练
# If loading a LoRA model
transformer = Transformer2DModel.from_pretrained("/mnt/workspace/qinshiyang/PixArt-alpha/test/diffuser/pre_20000_transformer", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "Your-LoRA-Model-Path")
pre_pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
# del transformer

pre_pipe.enable_model_cpu_offload()

pre_gen_images = pre_pipe(prompt).images


output_dir = "/mnt/workspace/qinshiyang/PixArt-alpha/test/diffuser_sample/match_compare_20000_6000"

# 调用函数合并图像并保存
merge_images_into_sheets(origin_images, pre_gen_images,finetune_gen_images, output_dir)
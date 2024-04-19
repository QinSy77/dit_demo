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


import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL,Transformer2DModel
# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too. #PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512  "PixArt-alpha/PixArt-XL-2-512x512"
#pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16, use_safetensors=True)

# If use DALL-E 3 Consistency Decoder
# pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

# If use SA-Solver sampler
# from diffusion.sa_solver_diffusers import SASolverScheduler
# pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')

# If loading a LoRA model
transformer = Transformer2DModel.from_pretrained("/mnt/workspace/qinshiyang/PixArt-alpha/test/diffuser/pre_20000_transformer", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "Your-LoRA-Model-Path")
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
# del transformer

# Enable memory optimizations.
pipe.enable_model_cpu_offload()

#A man in a suit and bow tie stands confidently against a black background.
#A young man with glasses and a black jacket is standing against a red wall, looking up and to the left.
#A man in a suit and tie, wearing a watch, sitting on a stool with a gray background.

#prompt = ["A man in a suit and bow tie stands confidently against a black background.","A young man with glasses and a black jacket is standing against a red wall, looking up and to the left."
#          ,"A man in a suit and tie, wearing a watch, sitting on a stool with a gray background.","A man in a suit and tie is drinking from a glass."]

#prompt = ["A young man with spiky hair and a snake print shirt is holding a black umbrella and drinking from a cup.","A young boy with spiky hair and a black martial arts uniform with an orange belt, standing in a stance with his hands clasped in front of him.",
#          "A man with dark hair and a brown jacket is smoking a cigarette, with a yellow exclamation point sticker in the bottom right corner.","A man in a black suit and tie, wearing glasses, standing against a black background."]

# prompt = ["A man in a black suit and tie, sitting with his hands clasped and a slight smile on his face.","A man in a red jersey with the number 24 on it, standing in front of a red background with white text.",
#           "A man with glasses and a suit, looking professional and confident.","A man in a blue suit with a red and blue striped tie, standing with his hands in his pockets."]


prompt = ["A man in a suit and bow tie stands confidently against a white background.","A young man with glasses and a black jacket is standing against a white wall,looking up and to the right."
          ,"A man in a suit and tie, wearing a watch, sitting on a stool with a gray background.","A man in a suit and tie is drinking from a glass."]

image = pipe(prompt).images

#for i in range(len(image)):
#    image[i].save(f"/mnt/workspace/qinshiyang/PixArt-alpha/test/diffuser_sample/man3_lora_400ep_512_test{i+1}.png")


combine_images_horizontally(image,"/mnt/workspace/qinshiyang/PixArt-alpha/test/diffuser_sample/pre_test2/man_test3_combine.png")  
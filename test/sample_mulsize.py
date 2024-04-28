import json
import os
# 假设你的原始文本文件名为 'original_data.txt'
input_file_path = 'original_data.txt'
# 输出文件名
output_file_path = 'filtered_data.txt'

# 给定的图像比例列表
desired_aspect_ratios = ['1_1', '3_4', '4_3', '9_16', '16_9', '2_3', '3_2']

# 初始化一个字典来存储每个比例的图像数据
selected_images = {ratio: [] for ratio in desired_aspect_ratios}


# 读取文本文件并处理每一行
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行为一个字典
        item = eval(line.strip())
        
        # 提取图片尺寸和比例
        height = item['img_sz'][0]
        width = item['img_sz'][1]
        ratio = item['face_ratio']
        
        # 提取图片路径
        img_path = item['img_path']
        
        # 检查图片文件是否存在
        if os.path.isfile(img_path):

            # 检查图像尺寸比例是否与给定的比例列表中的任一比例相匹配
            for ratio in desired_aspect_ratios:
                parts = ratio.split('_')
                target_aspect_ratio = int(parts[0]) / int(parts[1])
                
                # 计算图像的宽高比
                aspect_ratio = width  / height
                
                # 如果匹配并且该比例的图像数量还没有达到30张，则添加到selected_images中
                if aspect_ratio == target_aspect_ratio and len(selected_images[ratio]) < 30:
                    selected_images[ratio].append(line)
                    break  # 找到匹配的比例后，跳出循环继续处理下一条数据

# 将筛选出的图像数据写入新文件
with open(output_file_path, 'w') as output_file:
    for ratio in desired_aspect_ratios:
        for image_data in selected_images[ratio]:
            output_file.write(image_data)

print(f'每个尺寸比例抽取30张图像的数据已保存到 {output_file_path}')
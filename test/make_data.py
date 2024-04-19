import os
import json

# 假设文本文件的路径是 'image_info.txt'
file_path = '/mnt/workspace/qinshiyang/PixArt-alpha/test/male_renadomsz_210.txt'
# 指定JSON文件的绝对路径
absolute_json_name = '/mnt/workspace/qinshiyang/PixArt-alpha/test/210data_sigma/partition/210data_info.json'
data_info = []

# 读取文本文件并处理每一行
with open(file_path, 'r', encoding='utf-8') as file:
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
      
            # 添加到data_info列表中
            data_info.append({
                "height": height,
                "width": width,
                "ratio": ratio,
                "path": img_path,
                "prompt": item['img_ram']['internvl_cap']
            })
        


# 将data_info列表写入到JSON文件中
with open(absolute_json_name, "w", encoding='utf-8') as json_file:
    json.dump(data_info, json_file, ensure_ascii=False, indent=4)

# 打印完成信息
print(f"Data has been successfully saved to {absolute_json_name}")
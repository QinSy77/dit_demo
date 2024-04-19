
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
images,prompts=read_promt(filename,20)

# 输出结果
print(prompts)
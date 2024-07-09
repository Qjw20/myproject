import os
 
# 设置文件夹路径
folder_path = '/opt/data/private/data/blueberry2/images'
contents = []

 
# 获取文件夹中所有的文件和文件夹
images = os.listdir(folder_path)
 
# 遍历所有内容
for image in images:
    # 获取完整的路径
    path = os.path.join(folder_path, image)
    
    # 检查是否是文件
    if os.path.isfile(path):
        file_name = os.path.basename(path)
        content = file_name + " blueberry"
        print(content)
        contents.append(content)

# 打开文件进行写入
with open('./images_class.txt', 'w') as file:
    # 逐行写入内容
    for line in contents:
        file.write(line)
        file.write('\n')  # 添加换行符


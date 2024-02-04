import os
import pandas as pd
import re
'''获取文件夹里的子文件夹'''


def search_dir(folder_path):

    for root, dirs, files in os.walk(folder_path):
        if dirs:
            # os.walk乱序输出，先排序
            dirs.sort()
            # print('当前路径:' + root)
            # print('当前路径下所有子目录：', dirs)
            # 不再遍历子目录
            return dirs


TAG = "HOS"
ori_path = r'G:\EEGPT\EEG_Data\2023_JiNengSai_DATA\\'+TAG

# 一级子目录
list_1 = []
for i in search_dir(ori_path):
    list_1.append(os.path.join(ori_path,i))

# # 二级子目录
list_2 =[]
for i in list_1:
    for j in search_dir(i):
        list_2.append(os.path.join(i,j))


def key_func(folder):
    # 提取文件夹名称中的数字部分
    match = re.findall(r"(\d+)", folder)
    if len(match)>=2:
        folder_number = int(match[2])
        # print(match)
        return folder_number
    else:
        # 如果无法提取数字，则返回一个足够大的数确保文件夹被排序到最后
        return float("inf")

list_2 = sorted(list_2,key=key_func)

df = pd.DataFrame(data= list_2,columns = None)
# PATH为导出文件的路径和文件名
df.to_csv("file_list_{}.csv".format(TAG))
print(list_2)
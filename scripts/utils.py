"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 工具库
"""
import os
from tqdm import tqdm
import pandas as pd
from glob import glob


def generate_desc_csv(root_folder):
    """
    根据数据集路径生成数据集说明的csv文件
    :param root_folder:
    :return:
    """
    # 字符编码为数值
    name2label = {}  # "sq...":0
    for name in sorted(os.listdir(os.path.join(root_folder))):
        if not os.path.isdir(os.path.join(root_folder, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())
    file_id = []
    label = []
    for category in tqdm(os.listdir(root_folder)):
        images = glob(os.path.join(root_folder, category)+'/*')
        for img in images:
            file_id.append(img)
            label.append(name2label[category])

    df_desc = pd.DataFrame({'file_id': file_id, 'label': label})
    df_desc.to_csv("../data/desc.csv", encoding="utf8", index=False)


if __name__ == '__main__':
    generate_desc_csv("../data/Caltech101")



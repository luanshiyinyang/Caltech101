"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 工具库
"""


def generate_desc_csv(root_folder):
    """
    根据数据集路径生成数据集说明的csv文件
    :param root_folder:
    :return:
    """
    import os
    from tqdm import tqdm
    import pandas as pd
    from glob import glob
    # 字符编码为数值
    name2label = {}
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
            file_id.append(img.replace("\\", "/"))  # 契合linux平台
            label.append(name2label[category])

    df_desc = pd.DataFrame({'file_id': file_id, 'label': label})
    df_desc.to_csv("../data/desc.csv", encoding="utf8", index=False)


def save_pickle(python_object, saved_path):
    """
    保存Python对象为pickle文件
    :param python_object:
    :param saved_path:
    :return:
    """
    import pickle
    output = open(saved_path, 'wb')
    pickle.dump(python_object, output)
    output.close()


def load_pickle(file_path):
    """
    加载本地的pickle文件为Python对象
    :param file_path:
    :return:
    """
    import pickle
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


if __name__ == '__main__':
    generate_desc_csv("../data/Caltech101")



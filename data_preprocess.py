#读取txt文件，将每一类的图片放入以其对应标签命名的文件夹
import os
import shutil
import random

def organize_images(txt_file, output_dir):
    """
    根据txt文件中的标签，将图片移动到对应的文件夹。
    
    :param txt_file: 包含图片路径和标签的txt文件路径
    :param output_dir: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取txt文件
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 分割路径和标签
        image_path, label = line.strip().split()

        image_path="C:/Users/16494/Desktop/cat12/cat_data_sets_models/data_sets/cat_12/"+image_path

        # 创建以标签命名的文件夹
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # 移动图片到对应文件夹
        try:
            shutil.move(image_path, label_dir)
            print(f"Moved {image_path} to {label_dir}")
        except Exception as e:
            print(f"Error moving {image_path}: {e}")


def split_dataset(dataset_dir, output_dir, train_ratio=0.8):
    """
    将一个以类别文件夹组织的图像数据集分为训练集和测试集。
    
    :param dataset_dir: 原始数据集目录
    :param output_dir: 输出目录（包含 train 和 test 文件夹）
    :param train_ratio: 训练集所占比例（默认 80%）
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历每个类别文件夹
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # 创建类别文件夹
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # 获取所有图片路径
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        random.shuffle(images)

        # 按比例划分训练集和测试集
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # 移动图片
        for img_path in train_images:
            shutil.copy(img_path, train_class_dir)

        for img_path in test_images:
            shutil.copy(img_path, test_class_dir)

        print(f"Class '{class_name}': {len(train_images)} train, {len(test_images)} test")



def organize_test(folder_path):
    """
    列出文件夹中的所有文件名（不包含子文件夹中的文件）。
    
    :param folder_path: 文件夹路径
    :return: 文件名列表
    """
    filelist = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    return filelist



if __name__ == '__main__':
    #使用示例
    txt_file = './cat_data_sets_models/data_sets/cat_12/train_list.txt'  # 包含图片路径和标签的txt文件
    mid_dir = './cat_data_sets_models/data_sets/mid/'  # 输出文件夹
    final_dir = './cat_data_sets_models/data_sets/final_cat12/'
    #将训练集存储为文件夹以标签命名的结果
    organize_images(txt_file, mid_dir)
    #划分测试集与训练集
    split_dataset(mid_dir, final_dir, train_ratio=0.8)

    
   
    


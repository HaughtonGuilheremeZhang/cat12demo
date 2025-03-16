import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from resnet_model.resnet import resnet18
import time

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # 获取每个样本预测的类别
    correct = (predicted == labels).sum().item()  # 计算正确预测的数量
    total = labels.size(0)  # 样本总数
    accuracy = correct / total  # 准确率
    return accuracy


def loader(path, train=True):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    if train == True:
        # 加载训练集
        train_dataset = datasets.ImageFolder(root=path, transform=transform)
        # 创建 DataLoader
        loader_l = DataLoader(train_dataset, batch_size=64, shuffle=True)

    else:  # 加载测试集
        test_dataset = datasets.ImageFolder(root=path, transform=transform)
        loader_l = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return loader_l

class run():
    def __init__(self, model, device, optimizer, num_epochs, criterion, train_dir, test_dir):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.train_loader = loader(train_dir, train=True)
        self.test_loader = loader(test_dir, train=False)

    def train(self):

        #将数据权重送往GPU
        self.model.to(device)

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            running_accuracy = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 计算准确率
                accuracy = calculate_accuracy(outputs, labels)
                running_loss += loss.item()  # 累加损失
                running_accuracy += accuracy  # 累加准确率

            avg_loss = running_loss / len(self.train_loader)
            avg_accuracy = running_accuracy / len(self.train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

            self.test(epoch)

    def test(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        accuracy=0
        loss=0
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)

            accuracy = calculate_accuracy(outputs, labels)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()  # 累加损失
            running_accuracy += accuracy  # 累加准确率

        save_dir = 'resnet'+f'Epoch [{epoch + 1}]'+'.pth'
        print('test_acc:', accuracy)

        if accuracy > 0.9:
            # 保存模型和优化器状态字典
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, './work_place/'+save_dir)




if __name__ == '__main__':
    #def train(is_training, logits, images, labels):
    # 数据集路径
    train_dir = './cat_data_sets_models/data_sets/final_cat12/train'
    test_dir = './cat_data_sets_models/data_sets/final_cat12/test'
    #WORKPLACE

    model = resnet18(num_classes=12, initial_channels=32)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # 优化器设为ADAM
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 批数为10
    num_epochs = 10
    # CrossEntropyLoss交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    runner = run(model, device, optimizer, num_epochs, criterion, train_dir, test_dir)
    runner.train()

    #train(model, train_loader, device, optimizer, num_epochs, criterion)

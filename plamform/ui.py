import gradio as gr
import torch
import sys
from torchvision import models, transforms


class ui_na():
    def __init__(self, ckpt_path, model_type):
    # 加载一个训练的模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载保存的模型
        self.ckpt_path = ckpt_path
        checkpoint = torch.load(self.ckpt_path)
        model = model_type(num_classes=12, initial_channels=32)
        model = model.to(self.device)
        # 加载保存的模型参数
        # 从保存的字典中提取出 model_state_dict 并加载到模型中
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 设置为评估模式
        self.model = model

    def predict_image(self, img):
        # 预处理图像
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
        ])

        img = transform(img).unsqueeze(0).to(self.device)  # 增加一个维度，并将图像移动到设备上

        # 5. 推理
        with torch.no_grad():  # 禁用梯度计算，提高推理速度
            outputs = self.model(img)  # 前向传播
            _, predicted_idx = torch.max(outputs, 1)  # 获取最大值对应的类别


        # 返回类别名称
        class_names = ['Abyssinian', 'Leopard_cat', 'Siamese2','Bombay','Kart',
                       'Egyptian','Maine_Coon','Thailand','Ragdoll','Grey','Siamese1','Sphinx']  # 实际类别名
        return class_names[predicted_idx.item()]

    def ui_run(self):
        iface = gr.Interface(
            fn=self.predict_image,
            inputs=gr.Image(type="pil"),  # 使用 PIL 格式的图像作为输入
            outputs=gr.Textbox(label="Predicted"),
            title="Cat12 Prediction with ResNet",
            description="Upload an image of a cat and get its predicted type using a ResNet model."
        )
        # 启动 Gradio 应用
        iface.launch(share=True)


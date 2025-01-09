# 这里是inference.py的完整代码
import torch
from PIL import Image
import numpy as np
from models import SRCNN
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt

def prepare_image(image_path, scale_factor=4):
    """准备输入图像"""
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 保存原始大小用于后续比较
    org_width = img.width
    org_height = img.height
    
    # 将图像转换为YCbCr颜色空间
    ycbcr = img.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    
    # 将Y通道转换为tensor
    y = ToTensor()(y).view(1, -1, y.size[1], y.size[0])
    
    return y, cb, cr, org_width, org_height

def predict(model, img_tensor):
    """使用模型进行预测"""
    with torch.no_grad():
        preds = model(img_tensor).clamp(0.0, 1.0)
    return preds

def post_process(y_pred, cb, cr, device='cpu'):
    """后处理，将预测结果转换回RGB图像"""
    out_img_y = ToPILImage()(y_pred[0].cpu())
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    
    # 合并通道
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

def visualize_results(original_img, bicubic_img, srcnn_img):
    """可视化对比结果"""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(131)
    plt.imshow(original_img)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(bicubic_img)
    plt.title('Bicubic')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(srcnn_img)
    plt.title('SRCNN')
    plt.axis('off')
    
    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('/content/drive/MyDrive/SRCNN/weight/x4/best.pth'))  # 替换为你的模型路径
    model.eval()
    
    # 处理图像
    image_path = '/content/drive/MyDrive/SRCNN/testImage/lena.png'  # 替换为你的图像路径
    img = Image.open(image_path).convert('RGB')
    
    # 准备低分辨率输入
    y, cb, cr, original_width, original_height = prepare_image(image_path)
    y = y.to(device)
    
    # 进行预测
    y_pred = predict(model, y)
    
    # 后处理
    srcnn_img = post_process(y_pred, cb, cr)
    
    # 对比用的双三次插值结果
    bicubic_img = img.resize((original_width, original_height), Image.BICUBIC)
    
    # 可视化结果
    visualize_results(img, bicubic_img, srcnn_img)
    
    # 保存结果
    srcnn_img.save('srcnn_result.png')

if __name__ == '__main__':
    main()

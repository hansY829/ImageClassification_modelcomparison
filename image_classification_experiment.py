import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# 1. 加载模型函数
def load_models():
    """
    加载预训练的 AlexNet, VGG16 和 ResNet50 模型。
    使用 PyTorch 推荐的 Weights API 来获取最新的预训练权重。
    """
    print("正在加载模型...")
    # 获取 ImageNet 预训练权重枚举
    alexnet_weights = models.AlexNet_Weights.IMAGENET1K_V1
    vgg16_weights = models.VGG16_Weights.IMAGENET1K_V1
    resnet50_weights = models.ResNet50_Weights.IMAGENET1K_V1

    # 初始化模型并加载权重
    alexnet = models.alexnet(weights=alexnet_weights)
    vgg16 = models.vgg16(weights=vgg16_weights)
    resnet50 = models.resnet50(weights=resnet50_weights)
    
    # 关闭 Dropout 和 BatchNorm 的训练行为
    alexnet.eval()
    vgg16.eval()
    resnet50.eval()
    
    # 返回模型及其对应的权重信息
    return (alexnet, alexnet_weights), (vgg16, vgg16_weights), (resnet50, resnet50_weights)

# 2. 打印网络结构与参数对比函数
def analyze_models(alexnet_info, vgg16_info, resnet50_info):
    """
    对比三个模型的深度、参数量以及核心结构特点。
    """
    models_data = [
        ("AlexNet", alexnet_info[0], "8层", "5个卷积层 + 3个全连接层"),
        ("VGG16", vgg16_info[0], "16层", "13个卷积层 + 3个全连接层"),
        ("ResNet50", resnet50_info[0], "50层", "残差连接")
    ]
    #得益于-残差结构和全局平均池化层
    print("\n" + "="*80)
    print(f"{'模型名称':<15} | {'层数':<10} | {'总参数量':<15} | {'结构特点'}")
    print("-" * 80)
    
    for name, model, depth, features in models_data:
        # 计算所有需要梯度的参数总量
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<15} | {depth:<10} | {params:>14,} | {features}")
    print("="*80)

# 3. 生成条形图对比函数
def generate_bar_chart_comparison(image_urls, model_infos):
    """
    对多张图像进行推理，并将各模型的预测类别及其置信度以条形图形式展示。
    """
    num_images = len(image_urls)
    fig, axes = plt.subplots(num_images, 4, figsize=(24, 5 * num_images))
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    for i, url in enumerate(image_urls):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception:
            axes[i, 0].text(0.5, 0.5, "图像加载失败", ha='center', va='center')
            for j in range(4): axes[i, j].axis('off')
            continue

        # 第一列：显示原始图像
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"原始图像 {i+1}")
        axes[i, 0].axis('off')

        # 后三列：分别显示三个模型的预测置信度
        for j, (name, model, weights, _) in enumerate(model_infos):
            # 将图像转换为 0-1 之间的 float32 格式并进行标准化预处理
            rgb_img_float = np.float32(img) / 255
            input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            # 模型推理
            with torch.no_grad():
                output = model(input_tensor)
            
            # 计算 Softmax 概率并提取最高概率的类别
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            label = weights.meta["categories"][top_catid[0].item()]
            conf = top_prob[0].item()

            # 绘制柱状图
            axes[i, j+1].bar([label], [conf], color='skyblue')
            axes[i, j+1].set_ylim(0, 1) # y轴固定为 0-1 置信度范围
            axes[i, j+1].set_title(f"{name}\n预测: {label} ({conf:.2f})")
            # 旋转标签以防长类别名重叠
            for tick in axes[i, j+1].get_xticklabels(): tick.set_rotation(15)

    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    print("\n条形图对比已更新至 prediction_comparison.png")

# 4. 推理与 Grad-CAM 可视化函数
def run_inference_and_visualize(image_urls, model_infos):
    """
    利用 Grad-CAM 技术生成热力图，展示模型在分类时关注的图像区域。
    """
    num_images = len(image_urls)
    fig, axes = plt.subplots(num_images, 4, figsize=(24, 5 * num_images))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    for i, url in enumerate(image_urls):
        print(f"\n正在处理图像 {i+1}/{num_images}: {url.split('/')[-1].split('?')[0]}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            rgb_img = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
            rgb_img_float = np.float32(rgb_img) / 255
        except Exception as e:
            print(f"  无法处理图像: {e}")
            axes[i, 0].text(0.5, 0.5, "图像加载失败", ha='center', va='center')
            for j in range(4): axes[i, j].axis('off')
            continue

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"原始图像 {i+1}")
        axes[i, 0].axis('off')
        
        for j, (name, model, weights, target_layer) in enumerate(model_infos):
            # 预处理
            input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 初始化 Grad-CAM 对象，目标层通常选择模型最后一个卷积层
            # 最后一个卷积层包含了最丰富的空间和语义特征信息
            cam = GradCAM(model=model, target_layers=[target_layer])
            # 生成热力图数据
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]

            # 将热力图叠加到原始彩色图像上
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            
            # 推理以获取文字标签
            with torch.no_grad():
                output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            label = weights.meta["categories"][top_catid[0].item()]
            conf = top_prob[0].item()
            
            print(f"  - {name:<10} | 预测: {label:<20} | 置信度: {conf:.4f}")
            # 显示叠加了热力图的结果
            axes[i, j+1].imshow(visualization)
            axes[i, j+1].set_title(f"{name}\n预测: {label} ({conf:.2f})")
            axes[i, j+1].axis('off')
            
    plt.tight_layout()
    plt.savefig('final_heatmap_comparison.png')
    print("\n\n高质量热力图已保存至 final_heatmap_comparison.png")

# 5. 主程序入口
def main():
    # 扩展测试图像列表，包含：正常猫狗、容易混淆的品种、以及语义干扰样本（雕像）
    test_images = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?auto=format&fit=crop&w=400&q=80",  # 正常猫
        "https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=400&q=80",  # 正常狗
        "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&w=400&q=80",  # 缅因猫 (外形较粗犷，容易与野外猫科混淆)
        "https://images.unsplash.com/photo-1591160690555-5debfba289f0?auto=format&fit=crop&w=400&q=80",  # 博美犬 (毛发浓密，常被误认为金毛幼犬)
        "https://images.unsplash.com/photo-1590420485404-f86d22b8abf8?auto=format&fit=crop&w=400&q=80",  # 狼的雕像 (考验模型区分生物与非生物的能力)
        "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?auto=format&fit=crop&w=400&q=80"   # 跑车
    ]
    
    # 加载模型
    alexnet_info, vgg16_info, resnet50_info = load_models()
    # 分析参数
    analyze_models(alexnet_info, vgg16_info, resnet50_info)
    
    # 整合模型信息：名称、实例、权重、以及用于 Grad-CAM 的目标层
    # 对于 AlexNet 和 VGG，选择 features 序列中的最后一个卷积层
    # 对于 ResNet，选择最后一个 layer4 中的最后一个 Bottleneck 块
    model_infos = [
        ("AlexNet", alexnet_info[0], alexnet_info[1], alexnet_info[0].features[10]),
        ("VGG16", vgg16_info[0], vgg16_info[1], vgg16_info[0].features[28]),
        ("ResNet50", resnet50_info[0], resnet50_info[1], resnet50_info[0].layer4[-1])
    ]
    
    # 运行两种可视化流程
    run_inference_and_visualize(test_images, model_infos)
    generate_bar_chart_comparison(test_images, model_infos)

if __name__ == "__main__":
    main()

"""
图像预处理 Transform
提供标准化的图像预处理流程
"""
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Callable


def get_default_image_transform(image_size: int = 384) -> Callable:
    """
    获取默认的图像预处理 transform

    Args:
        image_size: 目标图像大小

    Returns:
        可调用的 transform
    """
    return T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_train_image_transform(image_size: int = 384,
                                augment: bool = True) -> Callable:
    """
    获取训练时的图像预处理 transform（可选数据增强）

    Args:
        image_size: 目标图像大小
        augment: 是否使用数据增强

    Returns:
        可调用的 transform
    """
    if augment:
        return T.Compose([
            T.RandomResizedCrop([image_size, image_size], scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return get_default_image_transform(image_size)


def get_inference_image_transform(image_size: int = 384) -> Callable:
    """
    获取推理时的图像预处理 transform（无数据增强）

    Args:
        image_size: 目标图像大小

    Returns:
        可调用的 transform
    """
    return T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def preprocess_image(image: Image.Image,
                     transform: Optional[Callable] = None,
                     image_size: int = 384) -> torch.Tensor:
    """
    预处理单个图像

    Args:
        image: PIL Image
        transform: 可选的自定义 transform
        image_size: 目标图像大小（如果 transform 未提供）

    Returns:
        [3, H, W] tensor
    """
    if transform is None:
        transform = get_default_image_transform(image_size)

    return transform(image)


def preprocess_batch(images: list,
                      transform: Optional[Callable] = None,
                      image_size: int = 384) -> torch.Tensor:
    """
    批量预处理图像

    Args:
        images: PIL Image 列表
        transform: 可选的自定义 transform
        image_size: 目标图像大小（如果 transform 未提供）

    Returns:
        [batch, 3, H, W] tensor
    """
    if transform is None:
        transform = get_default_image_transform(image_size)

    tensors = [transform(img) for img in images]
    return torch.stack(tensors)


def denormalize_image(tensor: torch.Tensor,
                      mean: tuple = (0.5, 0.5, 0.5),
                      std: tuple = (0.5, 0.5, 0.5)) -> torch.Tensor:
    """
    反归一化图像张量（用于可视化）

    Args:
        tensor: [3, H, W] 归一化后的张量
        mean: 归一化时使用的均值
        std: 归一化时使用的标准差

    Returns:
        反归一化后的张量
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)

    return tensor * std + mean


if __name__ == '__main__':
    # 测试预处理
    print('='*50)
    print('Testing Image Preprocessing')
    print('='*50)

    # 创建测试图像
    test_image = Image.new('RGB', (512, 512), color='white')
    print(f'Original image size: {test_image.size}')

    # 测试默认 transform
    transform = get_default_image_transform()
    tensor = preprocess_image(test_image, transform)
    print(f'Processed tensor shape: {tensor.shape}')  # [3, 384, 384]
    print(f'Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]')

    # 测试训练 transform（带增强）
    train_transform = get_train_image_transform(augment=True)
    train_tensor = preprocess_image(test_image, train_transform)
    print(f'Train tensor shape: {train_tensor.shape}')

    # 测试批量预处理
    images = [test_image, test_image, test_image]
    batch = preprocess_batch(images, transform)
    print(f'Batch shape: {batch.shape}')  # [3, 3, 384, 384]

    print('\n✓ Preprocessing test passed!')

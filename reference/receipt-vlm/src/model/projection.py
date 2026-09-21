"""
MLP Projection 层
将视觉特征映射到 LLM 隐藏空间
2层 MLP + GELU (LLaVA 风格)
"""
import torch
import torch.nn as nn
from typing import Optional


class MLPProjection(nn.Module):
    """
    2层 MLP 投影层
    将视觉编码器的输出投影到 LLM 的隐藏空间

    结构: vision_dim -> intermediate_dim -> llm_hidden_dim
    """

    def __init__(self,
                 vision_dim: int,
                 llm_hidden_dim: int,
                 intermediate_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 out_norm: bool = True,
                 target_row_norm: float = 0.65,
                 pool: int = 1):
        """
        out_norm / target_row_norm 解决「视觉 embedding 尺度压垮文本」。

        实测（路线 A Stage2 训练完成后的权重）：
            文本 embedding   行范数中位 = 0.65
            projection 输出  行范数中位 = 6.67      ← 10.2 倍
        再叠加数量差异——576 个图像 patch vs 60~90 个文本 token——送进 LLM 的
        序列里 86% 的位置是范数超大的视觉向量，第一层 RMSNorm 归一化后文本被
        压没了。模型的最优策略变成「忽略视觉、只靠语言先验」，于是输出全 null
        的合法 JSON：训练 loss 降到 0.37，字段 F1 却是 0。

        LLaVA 的 projection 后面确实没有归一化，但它有 558K 图文对做对齐预训练，
        projection 能自己学到合适尺度；我们只有 8000 张票据，差 70 倍，必须显式约束。

        pool: 对 patch 序列做 pool² 倍平均池化，降低图像 token 占比。
        """
        super().__init__()

        if intermediate_dim is None:
            intermediate_dim = llm_hidden_dim * 2

        self.vision_dim = vision_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.intermediate_dim = intermediate_dim
        self.pool = max(1, int(pool))

        # 2层 MLP
        self.fc1 = nn.Linear(vision_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, llm_hidden_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LayerNorm 把每行归一到单位方差（行范数 ≈ sqrt(hidden)），再乘一个
        # 可学习标量拉到文本 embedding 的量级。标量可学习，训练中能自行微调，
        # 但起点是对的——这正是小数据量下 projection 学不出来的东西。
        self.out_norm = nn.LayerNorm(llm_hidden_dim) if out_norm else None
        if out_norm:
            self.out_scale = nn.Parameter(
                torch.tensor(float(target_row_norm) / (llm_hidden_dim ** 0.5)))
        else:
            self.register_parameter("out_scale", None)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            vision_features: [batch, num_patches, vision_dim] 视觉特征

        Returns:
            [batch, num_patches, llm_hidden_dim] 投影后的特征
        """
        v = vision_features
        if self.pool > 1:
            b, p, d = v.shape
            k = self.pool * self.pool
            if p % k == 0:
                v = v.reshape(b, p // k, k, d).mean(dim=2)

        x = self.fc1(v)  # [batch, num_patches, intermediate_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, num_patches, llm_hidden_dim]
        if self.out_norm is not None:
            x = self.out_norm(x) * self.out_scale

        return x

    @property
    def num_parameters(self) -> int:
        """返回参数数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # 测试投影层
    print('Testing MLP Projection...')

    vision_dim = 768
    llm_hidden_dim = 512

    projection = MLPProjection(vision_dim, llm_hidden_dim)
    print(f'Projection parameters: {projection.num_parameters:,}')

    # 测试前向传播
    batch_size = 2
    num_patches = 576
    vision_features = torch.randn(batch_size, num_patches, vision_dim)

    output = projection(vision_features)
    print(f'Input shape: {vision_features.shape}')
    print(f'Output shape: {output.shape}')

    assert output.shape == (batch_size, num_patches, llm_hidden_dim)
    print('Projection test passed!')

### [📚] 视频学习脚手架: 经典卷积神经网络架构 (AlexNet, VGG, ResNet)

### 一、核心内容大纲 (Core Content Outline)
-   **ImageNet 分类挑战赛 (ImageNet Classification Challenge) 的历史回顾**
    -   浅层网络时代 (Shallow Network Era) (2010-2011)
        -   错误率维持在 25-28% 区间。
        -   获胜系统并非基于神经网络，而是依赖于手工设计的特征 (Hand-designed Features) 和线性分类器 (Linear Classifiers)。
    -   深度学习的突破：AlexNet (2012)
        -   首次引入卷积神经网络 (Convolutional Neural Networks)，将错误率大幅降低至 16.4%。
        -   标志着计算机视觉研究的转折点。
    -   持续的进步 (2013-2017)
        -   **2013**: ZFNet (错误率 11.7%)。
        -   **2014**: VGG 与 GoogLeNet (错误率 7.3% 和 6.7%)。
        -   **2015**: ResNet (残差网络) (错误率 3.6%)，网络层数达到前所未有的 152 层。
        -   **2016**: 模型集成 (Model Ensembles) (错误率 3.0%)，通过融合多种顶级架构进一步提升性能。
        -   **2017**: SENet (错误率 2.3%)。
    -   挑战赛的终结
        -   2017 年后，官方的年度 ImageNet 挑战赛不再举办，相关竞赛转移至 Kaggle 等平台。

-   **经典架构详解 (Classic Architectures in Detail)**
    -   **AlexNet 架构**
        -   核心组件 (Core Components): 5 个卷积层，3 个全连接层，使用 ReLU 激活函数。
        -   历史背景: 受限于当时 GPU (GTX 580, 3GB 显存) 性能，模型被拆分到两块 GPU 上训练。
    -   **VGG 网络: 更深、更规整的设计**
        -   设计准则: 统一使用 3x3 卷积和 2x2 池化，每经过一次池化，通道数翻倍。
        -   核心洞见: 两个 3x3 卷积层的堆叠，其感受野 (receptive field) 与一个 5x5 卷积层相同，但参数更少，非线性更强。
        -   缺点: 参数量和计算量巨大，非常低效。
    -   **GoogLeNet: 关注效率**
        -   引入 Inception 模块，并行处理不同尺度的特征。
        -   使用全局平均池化代替全连接层，大幅减少参数。
    -   **ResNet (残差网络): 解决深度网络训练难题**
        -   问题: 简单的堆叠网络层数会导致性能退化 (degradation)，这并非过拟合，而是优化难题。
        -   解决方案: 引入残差块 (Residual Block) 和捷径连接 (Shortcut connection)，让网络更容易学习恒等映射 (identity mapping)。

-   **后 ResNet 时代的架构演进 (Post-ResNet Architectural Evolution)**
    -   **提升 ResNet: ResNeXt**
        -   引入 “基数” (Cardinality) 的概念，即并行路径的数量。
        -   通过分组卷积 (Grouped Convolution) 实现，在保持计算复杂度不变的情况下，通过增加分支数量提升了模型性能。
    -   **Squeeze-and-Excitation 网络 (SENet)**
        -   在 ResNeXt 的基础上，为每个残差块增加一个 "Squeeze-and-Excite" 分支。
        -   该分支通过全局池化 (Squeeze) 和两个全连接层 (Excite) 来学习通道间的依赖关系，动态地重新校准 (recalibrate) 特征图的通道权重，从而引入了全局上下文信息。
    -   **密集连接网络 (DenseNet - Densely Connected Neural Networks)**
        -   提出一种新的连接模式：密集块 (Dense Block)。
        -   在块内，每一层都与所有前面的层直接相连，其输入是前面所有层特征图的拼接 (Concatenation)。
        -   极大地促进了特征重用 (Feature Reuse) 和梯度流动。
    -   **高效轻量级网络 (Efficient & Tiny Networks)**
        -   目标: 为移动设备等资源受限场景设计网络。
        -   **MobileNets**:
            -   核心思想是使用深度可分离卷积 (Depthwise Separable Convolution) 来替代标准卷积。
            -   将标准卷积分解为一次深度卷积 (Depthwise Convolution) 和一次逐点卷积 (Pointwise Convolution)，从而大幅降低计算成本。
        -   **ShuffleNets**: 另一类高效网络架构。
    -   **神经架构搜索 (Neural Architecture Search - NAS)**
        -   目标: 自动化神经网络的设计过程。
        -   工作原理: 使用一个“控制器”网络 (Controller, 通常是 RNN) 来生成“子网络” (Child Network) 的架构描述。
        -   通过训练子网络并将其性能作为奖励信号，利用策略梯度 (Policy Gradient) 来更新控制器，使其能生成更好的架构。
        -   挑战: 过程极为昂贵，初代研究需要在 800 个 GPU 上训练 28 天。后续研究致力于提升搜索效率。

### 二、关键术语定义 (Key Term Definitions)
-   **AlexNet**: 2012 年 ImageNet 挑战赛的冠军模型，是一个深度卷积神经网络，它的成功标志着深度学习在计算机视觉领域的革命性突破。
-   **VGG 网络 (VGG Network)**: 2014 年 ImageNet 挑战赛的代表模型之一，特点是结构非常规整，完全由 3x3 的卷积层和 2x2 的池化层堆叠而成。
-   **GoogLeNet**: 2014 年 ImageNet 挑战赛的冠军模型，通过引入 Inception 模块，专注于提升计算效率。
-   **Inception 模块 (Inception Module)**: GoogLeNet 的核心构建块，在一个模块内并行使用不同尺寸的卷积核和池化操作，然后将结果拼接起来，以在不同尺度上提取特征。
-   **残差网络 (ResNet - Residual Networks)**: 2015 年 ImageNet 挑战赛的冠军模型，通过引入“残差块”和“捷径连接”解决了深度神经网络训练中的性能退化问题。
-   **ResNeXt**: ResNet 的一种改进，通过引入“基数” (Cardinality) 的概念并使用分组卷积 (Grouped Convolution) 增加了模型的表达能力，同时控制了计算复杂度。
-   **Squeeze-and-Excitation 网络 (SENet)**: 2017 年 ImageNet 挑战赛的冠军模型，通过一个小型子网络动态地学习各通道的权重，增强了网络的特征表达。
-   **密集连接网络 (DenseNet - Densely Connected Neural Networks)**: 一种网络架构，其核心的“密集块”中每一层都与该块内所有前面的层相连接，通过特征拼接实现信息最大化流动。
-   **深度可分离卷积 (Depthwise Separable Convolution)**: MobileNet 中使用的核心技术，将标准卷积分解为深度卷积 (per-channel spatial filtering) 和逐点卷积 (1x1 convolution for combining channels)，从而大幅降低计算量和参数量。
-   **神经架构搜索 (Neural Architecture Search - NAS)**: 一个自动化设计神经网络架构的领域，旨在通过算法（如强化学习）来搜索最优的网络结构，而不是依赖人工设计。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)
本次视频的核心是模型架构的设计理念，未直接包含可运行的代码片段。以下是几个关键架构块的逻辑示意：
-   **AlexNet 架构**:
    -   **输入**: 227x227x3
    -   **conv1**: 64 个 11x11 卷积核，步长 4，填充 2
    -   **pool1**: 3x3 最大池化，步长 2
    -   **conv2**: 192 个 5x5 卷积核，步长 1，填充 2
    -   **pool2**: 3x3 最大池化，步长 2
    -   **conv3**: 384 个 3x3 卷积核，步长 1，填充 1
    -   **conv4**: 256 个 3x3 卷积核，步长 1，填充 1
    -   **conv5**: 256 个 3x3 卷积核，步长 1，填充 1
    -   **pool5**: 3x3 最大池化，步长 2
    -   **flatten**: 将卷积层输出展平
    -   **fc6**: 全连接层，4096 个单元
    -   **fc7**: 全连接层，4096 个单元
    -   **fc8**: 全连接层，1000 个单元 (对应 ImageNet 类别数)

-   **VGG 架构设计原则**:
    1.  所有卷积层使用 3x3 卷积核，步长1，填充1。
    2.  所有最大池化层使用 2x2 池化核，步长2。
    3.  每经过一次池化，通道数翻倍。

-   **ResNet 的瓶颈残差块 ("Bottleneck" Residual block)**:
    1.  `Conv 1x1` (降维)
    2.  `Conv 3x3`
    3.  `Conv 1x1` (升维)
    4.  将输出与输入（通过捷径连接）相加。

-   **ResNeXt 块 (ResNeXt Block)**:
    -   将输入分成 `G` 个平行的低维分支（`G` 是基数）。
    -   对每个分支应用一系列相同的变换（如 ResNet 的瓶颈块）。
    -   将所有分支的结果相加。
    -   这在实现上等价于使用 `groups=G` 的分组卷积 (Grouped Convolution)。

-   **DenseNet 的密集块 (Dense Block)**:
    -   `Input` -> `Layer 1`
    -   `Input` & `Output of Layer 1` (Concatenated) -> `Layer 2`
    -   `Input` & `Output of Layer 1` & `Output of Layer 2` (Concatenated) -> `Layer 3`
    -   ... 以此类推

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   对于 AlexNet 的第一个卷积层 (conv1)，给定输入尺寸 (3x227x227)、滤波器数量 (64)、核尺寸 (11)、步长 (4) 和填充 (2)，它的输出尺寸 (Output size) 是多少？
    -   (紧接上一个问题) 输出的通道数 (C) 是多少？
    -   (紧接上一个问题) 输出特征图需要多少内存 (KB)？
    -   (紧接上一个问题) 这一层有多少可学习的参数 (params)？
    -   (紧接上一个问题) 完成这一层的前向传播需要多少次浮点运算 (flop)？
-   VGG 这个名字代表什么？ (What does VGG stand for?)
-   **我应该使用哪种架构？ (Which Architecture should I use?)**
    -   **讲师建议**: 不要当英雄 (Don't be a hero)。对大多数问题，应该使用现成的、经过验证的架构，而不是尝试设计自己的。
        -   如果只关心**准确率**：`ResNet-50` 或 `ResNet-101` 是很好的选择。
        -   如果想要一个**高效的网络** (用于实时应用、移动端等)：尝试 `MobileNets` 和 `ShuffleNets`。
        

---




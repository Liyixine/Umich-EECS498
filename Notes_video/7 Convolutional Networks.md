### [📚] 视频学习脚手架: CS231n Lecture 7 - Convolutional Neural Networks

### 一、核心内容大纲 (Core Content Outline)
-   **传统分类器的问题 (Problem with Traditional Classifiers)**
    -   线性分类器 (Linear Classifiers) 和全连接网络 (Fully-Connected Networks) 要求输入被“展平” (flattened) 为一维向量。
    -   这个过程会破坏图像固有的二维空间结构 (spatial structure)。
-   **解决方案：卷积神经网络 (Solution: Convolutional Neural Networks)**
    -   定义能够直接在图像（或具有空间结构的数据）上操作的新型计算节点（层）。
    -   核心思想是利用并尊重输入的空间结构。
-   **卷积网络 (ConvNet) 的组件 (Components of a ConvNet)**
    -   **回顾：全连接网络 (FCN) 的组件**
        -   全连接层 (Fully-Connected Layers)
        -   激活函数 (Activation Function)，如 ReLU。
    -   **新增：卷积网络的特有组件**
        -   **卷积层 (Convolution Layers)**
        -   **池化层 (Pooling Layers)**
        -   **归一化层 (Normalization Layers)**
-   **深入理解卷积层 (Convolution Layer in Depth)**
    -   **输入 (Input)**: 将图像视为一个三维体 (3D volume)，维度为 `通道数 x 高度 x 宽度` (Channels x Height x Width)。
    -   **滤波器/核 (Filter/Kernel)**:
        -   一系列小的、可学习的权重集合。
        -   滤波器在空间上是局部的，但其深度必须与输入体的深度完全匹配。
        -   示例：一个 `5x5` 的滤波器作用于一个 `3x32x32` 的输入图像，其真实尺寸为 `3x5x5`。
    -   **卷积操作 (Convolution Operation)**:
        1.  将滤波器在输入体的空间维度上（高度和宽度）滑动。
        2.  在每一个位置，计算滤波器与对应输入体小块之间的点积 (dot product)，并加上一个偏置项 (bias)。
        3.  每个位置的点积运算产生一个单一的数值。
    -   **激活图 (Activation Map)**:
        -   单个滤波器在整个输入体上滑动后，生成一个二维的激活图（也称为特征图, feature map）。
        -   激活图的尺寸取决于输入尺寸、滤波器尺寸、步长和填充。
    -   **输出体 (Output Volume)**:
        -   卷积层通常包含一“组” (bank) 独立的滤波器。
        -   每个滤波器产生一张独立的激活图。
        -   将所有激活图堆叠起来，形成最终的输出体。
        -   输出体的深度 (depth) 等于滤波器的数量。
-   **处理空间维度 (Handling Spatial Dimensions)**
    -   **问题：特征图收缩 (Feature Map Shrinking)**
        -   每次卷积操作都会导致输出的空间尺寸小于输入。
        -   这限制了网络的深度，因为特征图最终会消失。
    -   **解决方案：填充 (Padding)**
        -   在输入体的边界周围添加额外的像素，通常是零（零填充, zero-padding）。
        -   **"Same" Padding**: 一种常见的策略，设置填充大小 `P = (K - 1) / 2`（其中 K 是滤波器尺寸），以确保输出和输入的空间尺寸相同。
    -   **解决方案：步长 (Stride)**
        -   定义滤波器在输入体上滑动的步长。
        -   步长大于1可以实现下采样 (downsampling)，减小输出的空间尺寸。
    -   **输出尺寸公式**:
        -   `Output_size = (W - K + 2P) / S + 1`
        -   其中 `W` 是输入尺寸, `K` 是核尺寸, `P` 是填充, `S` 是步长。
-   **感受野 (Receptive Fields)**
    -   输出特征图中的一个元素，其值所依赖的输入图像区域。
    -   堆叠卷积层可以有效增大感受野。例如，两个 `3x3` 卷积层的堆叠，其感受野相当于一个 `5x5` 的卷积层。
-   **池化层 (Pooling Layers)**
    -   一种不含可学习参数的下采样方法。
    -   **最大池化 (Max Pooling)**: 在一个局部区域（如 2x2）内，取最大值作为输出。
    -   作用：
        -   逐步减小表示的空间尺寸。
        -   引入对小的空间平移的**不变性 (invariance)**。
-   **归一化层 (Normalization Layers)**
    -   **批归一化 (Batch Normalization)**:
        -   **动机**: 解决“内部协变量偏移 (internal covariate shift)”问题，稳定并加速深度网络的训练。
        -   **核心思想**: 对一个层的输出进行归一化，使其具有零均值和单位方差。
        -   **操作**:
            1.  在一个小批量 (mini-batch) 数据中，对每个特征通道独立计算均值和方差。
            2.  使用这些统计量对该通道的特征进行归一化。
            3.  引入可学习的缩放 (scale, γ) 和偏移 (shift, β) 参数，让网络可以恢复原始的表示能力。
        -   **训练与测试的区别**:
            -   **训练时**: 使用当前小批量的均值和方差。
            -   **测试时**: 使用在整个训练过程中累积的均值和方差的移动平均值。

### 二、关键术语定义 (Key Term Definitions)
-   **卷积神经网络 (Convolutional Neural Network, CNN/ConvNet)**: 一类深度神经网络，其设计核心是利用和尊重数据的空间结构，通过卷积层、池化层等操作来处理具有网格状拓扑结构的数据（如图像）。
-   **展平 (Flatten)**: 将多维数据（如图像矩阵）转换为一维向量的过程，这通常会丢失原始数据的空间信息。
-   **滤波器 / 核 (Filter / Kernel)**: 在卷积层中，一个小的、可学习的权重矩阵，它在输入数据上滑动，通过点积运算来检测特定的局部特征（如边缘、颜色、纹理）。
-   **卷积 (Convolution)**: 将滤波器（核）在输入数据上滑动，并在每个位置计算点积以生成特征图的过程。
-   **激活图 / 特征图 (Activation Map / Feature Map)**: 单个滤波器在整个输入上进行卷积操作后生成的二维输出。它表示该滤波器所检测的特征在输入图像中不同位置的响应强度。
-   **步长 (Stride)**: 滤波器在输入数据上每次滑动的像素数。
-   **填充 (Padding)**: 在输入数据的边界周围添加像素（通常为零）以控制输出空间尺寸的过程。
-   **感受野 (Receptive Field)**: 在卷积网络中，输出层的一个神经元所能“看到”的输入图像的区域大小。
-   **池化 (Pooling)**: 一种下采样技术，通过在局部区域内应用一个固定的函数（如取最大值或平均值）来减小特征图的空间维度。
-   **最大池化 (Max Pooling)**: 一种池化操作，在局部区域内选择最大值作为输出。
-   **批归一化 (Batch Normalization)**: 一种对层激活值进行归一化的技术，通过计算小批量数据的均值和方差来调整激活值分布，使其具有零均值和单位方差，从而加速和稳定训练过程。
-   **内部协变量偏移 (Internal Covariate Shift)**: 在深度网络训练过程中，由于前一层参数的更新导致后一层输入分布发生变化的现象。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)
本次视频未包含完整的代码示例，但详细讲解了核心操作的数学逻辑。

-   **卷积层输出尺寸计算 (Conv Layer Output Size Calculation)**:
    -   给定输入尺寸 `W x W`，滤波器尺寸 `K x K`，填充 `P`，步长 `S`。
    -   输出尺寸 `W' x W'` 的计算公式为：
    -   `W' = (W - K + 2P) / S + 1`
    -   该公式同样适用于高度 `H`。

-   **批归一化 (Batch Normalization)**:
    -   对于一个批量的激活值 `x`，其中包含 `N` 个样本，每个样本是 `D` 维。
    -   **1. 计算小批量均值 (μ) 和方差 (σ^2)** (对每个特征维度 `j`):
        -   `μ_j = (1/N) * Σ(x_{i,j})` for i=1 to N
        -   `σ_j^2 = (1/N) * Σ(x_{i,j} - μ_j)^2` for i=1 to N
    -   **2. 归一化 (Normalize)**:
        -   `x̂_{i,j} = (x_{i,j} - μ_j) / sqrt(σ_j^2 + ε)` (ε 是为了防止除零的微小常数)
    -   **3. 缩放和偏移 (Scale and Shift)**:
        -   `y_{i,j} = γ_j * x̂_{i,j} + β_j`
        -   其中 `γ` (gamma) 和 `β` (beta) 是可学习的参数。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   如果我们堆叠两个卷积层会发生什么？(What happens if we stack two convolution layers?)
-   (隐含问题) 为什么我们需要非线性激活函数（如ReLU）穿插在卷积层之间？
-   (隐含问题) 对于大型图像，我们需要多少层才能让每个输出“看到”整个图像？
-   (隐含问题) 为什么批归一化 (Batch Normalization) 在测试时需要与训练时有不同的行为？

---

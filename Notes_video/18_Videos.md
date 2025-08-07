### [📚] 视频学习脚手架: Lecture 18: Videos

### 一、核心内容大纲 (Core Content Outline)
-   **引言与回顾 (Introduction and Recap)**
    -   回顾：计算机视觉任务中的2D识别 (2D Recognition)
        -   图像分类 (Classification)
        -   语义分割 (Semantic Segmentation)
        -   物体检测 (Object Detection)
        -   实例分割 (Instance Segmentation)
    -   回顾：上一讲的3D形状 (3D Shapes)
        -   从单张图像预测3D形状
        -   处理3D输入数据
        -   3D形状的表示方法：深度图 (Depth Map), 体素网格 (Voxel Grid), 隐式表面 (Implicit Surface), 点云 (Pointcloud), 网格 (Mesh)
-   **今日主题：视频处理 (Today's Topic: Videos)**
    -   **视频的基本表示 (Basic Representation of Videos)**
        -   核心思想：视频 = 2D + 时间 (Time)
        -   视频是图像的序列 (a sequence of images)
        -   表示为4D张量 (4D Tensor): `T x 3 x H x W` 或 `3 x T x H x W` (T代表时间维度)
    -   **视频分析任务：视频分类 (Video Task: Video Classification)**
        -   示例：输入一段视频，输出动作类别（如：跑步 Running）。
        -   与图像识别的核心区别：
            -   图像识别：通常识别**物体 (objects)** (名词 nouns)
            -   视频识别：通常识别**动作 (actions)** (动词 verbs)
-   **视频数据的挑战与解决方案 (Challenges & Solutions with Video Data)**
    -   **问题：视频数据量巨大 (Problem: Videos are big!)**
        -   视频通常约为每秒30帧 (30 frames per second, fps)
        -   未压缩视频的尺寸估算：
            -   标清 (SD, 640x480): 约 1.5 GB/分钟
            -   高清 (HD, 1920x1080): 约 10 GB/分钟
    -   **解决方案：在视频片段上进行训练 (Solution: Train on short clips)**
        -   降低帧率 (Low fps) 和空间分辨率 (low spatial resolution)
        -   示例：从原始长视频中采样出3.2秒、5fps、112x112分辨率的短片进行处理。
        -   训练策略：从长视频中随机采样短片 (clips) 进行训练。
        -   测试策略：在测试视频上采样多个短片，对所有短片的预测结果进行平均。
-   **视频分类的架构 (Architectures for Video Classification)**
    -   **1. 单帧CNN (Single-Frame CNN)**
        -   简单思路：训练一个标准的2D CNN，独立地对视频的每一帧进行分类。
        -   在测试时，对所有帧的预测概率进行平均。
        -   这是一个**非常强大**的基线模型 (strong baseline)，应始终作为首选尝试。
    -   **2. 晚期融合 (Late Fusion)**
        -   直觉：获取每一帧的高层级外观特征，然后将它们组合起来。
        -   实现方式一 (with FC layers):
            1.  对每一帧独立运行2D CNN提取特征。
            2.  将所有帧的特征图展平 (Flatten) 并连接 (concatenate) 成一个巨大的向量。
            3.  将该向量送入一个多层感知机 (MLP) 进行分类。
        -   实现方式二 (with pooling):
            1.  对每一帧独立运行2D CNN提取特征。
            2.  在时间和空间维度上进行平均池化 (Average Pool over space and time)。
            3.  将池化后的特征向量送入一个线性层 (Linear) 进行分类。
        -   问题：难以比较帧与帧之间的低层级运动，因为时间信息的融合发生得太晚。
    -   **3. 早期融合 (Early Fusion)**
        -   直觉：在网络的第一层卷积就比较不同的帧。
        -   实现方式：将输入 `T x 3 x H x W` 重塑 (reshape) 为 `(3T) x H x W`，即将时间维度堆叠到通道维度上。
        -   之后，整个网络就是一个标准的2D CNN。
        -   问题：缺乏时间上的平移不变性 (temporal shift-invariance)。模型需要为发生在不同时间点的相同运动学习不同的滤波器。
    -   **4. 3D CNN**
        -   直觉：使用3D版本的卷积和池化，在整个网络中**缓慢地融合 (slowly fuse)** 时间信息。
        -   架构：
            -   网络的每一层都是一个4D张量 (`C_in x T x H x W` 输入, `C_out x T x H x W` 输出)。
            -   使用3D卷积核 (Weight: `C_out x C_in x 3x3x3`) 和3D池化操作。
            -   滤波器在空间（x, y）和时间 (t) 维度上滑动。
        -   优势：由于滤波器在时间维度上滑动，因此具有**时间平移不变性**。可以学习识别在不同时间点发生的相同动作。
        -   可视化：3D CNN的第一层滤波器可以被看作是小型的视频片段，学习检测时空模式（如运动和颜色变化）。
-   **架构对比与总结 (Architecture Comparison & Summary)**
    -   **Sports-1M 数据集 (Sports-1M Dataset)** 上的Top-5准确率对比：
        -   数据集：1百万 YouTube 视频，标注了487种不同运动类型。包含精细化区分（例如：马拉松 vs 超级马拉松）。
        -   结果：
            -   单帧 (Single Frame): 77.7%
            -   早期融合 (Early Fusion): 76.8%
            -   晚期融合 (Late Fusion): 78.7%
            -   3D CNN: **80.2%**
        -   结论：3D CNN表现最好，但单帧模型作为一个简单的基线，效果也出奇地好。
        -   注意：原始论文来自2014年，模型在CPU集群上训练耗时一个月。自2014年以来，3D CNNs 的性能已大幅提升。
    -   **C3D：3D CNN 的 VGG (C3D: The VGG of 3D CNNs)**
        -   一种3D CNN，全部使用3x3x3卷积和2x2x2池化（除了第一个池化层是1x2x2）。
        -   发布了在 Sports-1M 上预训练的模型，许多人将其用作视频特征提取器。
        -   问题：3x3x3卷积的计算成本非常高！
            -   AlexNet (图像，0.7 GFLOP)
            -   VGG-16 (图像，13.6 GFLOP)
            -   C3D (视频，39.5 GFLOP) - 是 VGG 的2.9倍！即使输入很小 (3x16x112x112)。
        -   性能：C3D 在 Sports-1M 数据集上的 Top-5 准确率达到 **84.4%**。
        -   总结：与图像识别类似，更大、更深、计算成本更高的模型，在视频识别中也能带来更高的准确率。
-   **视频模型中的时空处理 (Spatio-Temporal Processing in Video Models)**
    -   **从运动中识别动作 (Recognizing Actions from Motion)**
        -   人类可以仅凭运动信息轻松识别动作。例如，仅通过移动的点（如关节标记点）就能识别出人走路、爬行、骑独轮车等动作。
        -   这表明人类大脑可能对运动和视觉外观进行不同的处理。
    -   **测量运动：光流 (Measuring Motion: Optical Flow)**
        -   光流：提供图像 `I_t` 和 `I_{t+1}` 之间的位移场 `F(x, y) = (dx, dy)`，表示每个像素在下一帧中的移动位置。
        -   `I_{t+1}(x+dx, y+dy) = I_t(x,y)`。
        -   光流可以突出局部运动信息，是提供给CNNs 的低级运动信号。
    -   **分离运动和外观：双流网络 (Separating Motion and Appearance: Two-Stream Networks)**
        -   架构：包含两个并行的卷积神经网络流。
            -   **空间流 (Spatial stream ConvNet)**：输入是**单帧图像** (`3 x H x W`)，处理视频的**外观 (appearance)** 信息。
            -   **时间流 (Temporal stream ConvNet)**：输入是**多帧光流的堆叠** (`2 * (T-1) x H x W`)，处理视频的**运动 (motion)** 信息。
                -   `2 * (T-1)`：表示 `T-1` 对帧的光流，每对光流包含 `dx` 和 `dy` 两个通道。
                -   在时间流的第一层卷积中进行**早期融合 (Early Fusion)**，处理所有光流图像。
        -   最终**类别分数融合 (class score fusion)**：通过对两个流的预测分数（如Softmax输出）进行平均来融合。
        -   UCF-101 数据集上的准确率：
            -   3D CNN: 65.4%
            -   空间流 (Spatial only): 73%
            -   时间流 (Temporal only): **83.7%** (运动信息非常重要!)
            -   双流 (Two-stream, fuse by average): 86.9%
            -   双流 (Two-stream, fuse by SVM): **88%**
        -   结论：时间流比空间流在动作识别中更为强大，融合两者能进一步提高性能。
-   **建模长期时间结构 (Modeling Long-Term Temporal Structure)**
    -   目前为止的3D CNNs 和双流网络主要关注短视频片段中的局部运动 (约2-5秒)。如何处理长期时间结构？
    -   **想法**：提取每一帧或每个短片段的局部特征（使用2D或3D CNN），然后使用**循环神经网络 (Recurrent Neural Network, RNN)** 来处理这些局部特征序列。
        -   可以实现**多对一 (Many-to-one)** 模式（视频末尾一个输出），或**多对多 (Many-to-many)** 模式（每帧一个输出）。
        -   早期工作 (2011年) 已提出结合3D CNNs 和 LSTMs 的架构。
    -   **循环卷积网络 (Recurrent Convolutional Network)**
        -   核心思想：将标准RNN中的矩阵乘法替换为2D卷积操作。
        -   网络中的每个特征图 (`C x H x W`) 都依赖于两个输入：
            1.  同一层的前一个时间步的特征图（时间依赖）。
            2.  前一个层级的当前时间步的特征图（空间依赖）。
        -   这种结构在时间和空间维度上共享权重。
        -   可以应用于RNNs 的各种变体（如GRU, LSTM）。
    -   **RNN 的问题**：
        -   RNN 处理长序列时速度慢，无法并行化 (sequential computation)。
        -   与1D卷积相比：1D卷积可以并行化，但要“看到”整个长序列需要堆叠非常多的卷积层。3D卷积类似于将1D卷积应用于3D网格。
    -   **时空自注意力 (Spatio-Temporal Self-Attention) (非局部块 Nonlocal Block)**
        -   自注意力机制 (Self-Attention)：
            -   输入：一组向量。
            -   优势：善于处理长序列（一层后输出能“看到”所有输入），高度并行化。
            -   劣势：内存密集型。
        -   应用于视频：
            -   从3D CNN 中获取特征 (`C x T x H x W`)。
            -   将这些特征视为一组 `T * H * W` 个 C 维向量。
            -   通过1x1x1卷积计算 Queries (Q), Keys (K), Values (V)。
            -   计算注意力权重（`transpose(Q)` * `K`，然后 Softmax），得到一个 `(THW) x (THW)` 的亲和矩阵 (affinity matrix)。
            -   输出：Values 的加权和，权重由亲和矩阵给出。
        -   **非局部块 (Nonlocal Block)**：一个可以插入到现有3D CNN架构中的模块。
            -   **技巧 (Trick)**：初始化非局部块中最后一个卷积层的所有权重为0。这样整个块在初始化时计算的是恒等函数 (identity function)，允许将其插入到预训练的3D CNN中，并通过微调来逐步学习非局部依赖。
            -   优势：能够引入**全局时间处理 (global temporal processing)**，克服了传统3D CNN局部感受野的限制。
-   **膨胀2D网络到3D (Inflating 2D Networks to 3D, I3D)**
    -   核心思想：复用图像领域已验证的优秀2D CNN 架构来处理视频。
    -   方法：将每个2D卷积/池化层 `K_h x K_w` 替换为3D版本 `K_t x K_h x K_w`。
    -   **权重初始化技巧 (Weight Initialization Trick)**：
        -   将2D卷积核的权重在时间维度上复制 `K_t` 次，并除以 `K_t`。
        -   这样，当输入视频是“恒定”的（所有帧都相同）时，3D卷积的输出与原始2D卷积的输出完全相同。
    -   **性能表现 (Top-1 Accuracy on Kinetics-400 dataset)** (Wang et al., CVPR 2017)
        -   数据集：Kinetics-400 (大规模动作识别数据集)。
        -   Per-frame CNN：57.9% (从头训练), 62.2% (ImageNet 预训练)
        -   CNN+LSTM：63.3% (从头训练), 62.8% (ImageNet 预训练)
        -   Two-stream CNN：65.6% (从头训练), 68.4% (ImageNet 预训练)
        -   膨胀CNN (Inflated CNN)：71.1% (从头训练), **71.6%** (ImageNet 预训练) - 显著提升。
        -   双流膨胀CNN (Two-stream inflated CNN)：**74.2%** (ImageNet 预训练) - 最优性能。
-   **视频模型可视化 (Visualizing Video Models)**
    -   方法：与图像模型可视化类似，通过梯度上升寻找能最大化特定类别分数的输入（图像和光流）。
    -   可以添加项来鼓励光流的空间平滑性 (spatially smooth flow)，并调整惩罚项以区分“慢速”与“快速”运动。
    -   **示例：应用眼部化妆 (Apply Eye Makeup)**
        -   外观 (Appearance) 图像：看起来是多张人脸的叠加（模型寻找人脸作为基础）。
        -   “慢速”运动 ("Slow" motion) 光流：看起来像头部或手部的轻微运动（模型关注较慢的动作）。
        -   “快速”运动 ("Fast" motion) 光流：看起来像刷子在眼部快速移动的局部动作（模型关注快速、精细的动作）。
        -   证明：模型能够从外观和不同速度的运动中学习识别复杂动作。
-   **处理时空差异：慢快网络 (Treating Time and Space Differently: SlowFast Networks) (当前最先进模型 SoTA)**
    -   核心理念：并非所有特征都需要相同的时间采样率。
    -   架构：两个并行的网络路径。
        -   **慢路径 (Slow Pathway)**：以**低帧率 (low framerate)** 运行（例如，原始视频的1/16帧），但使用**大量通道 (many channels)**。主要处理视频的**外观和语义 (appearance and semantics)** 信息。
        -   **快路径 (Fast Pathway)**：以**高帧率 (high framerate)** 运行（例如，原始视频的1/2帧），但使用**少量通道 (few channels)**，计算量很小（<20%的总计算量）。主要处理视频的**快速运动 (fast motion)** 信息。
    -   **侧向连接 (Lateral connections)**：将快路径中的信息融合回慢路径的多个阶段，使慢路径也能获取细粒度的运动信息。
    -   **实现**：通常以 ResNet-50 作为骨干网络，通过膨胀 (I3D) 来构建时空特征提取器。
    -   **优势**：这种方法能有效结合不同时间尺度的信息，是目前视频理解领域最先进的架构之一。
-   **超越分类：其他视频任务 (Beyond Classification: Other Video Tasks)**
    -   **时序动作定位 (Temporal Action Localization)**：给定一个长且未经剪辑的视频序列，识别出对应不同动作的时间段（帧区间）。
        -   任务：先生成时序提议 (temporal proposals)，然后对这些提议进行分类。
        -   架构：类似于 Faster R-CNN，但操作在时间维度上。
    -   **时空检测 (Spatio-Temporal Detection)**：给定一个长且未经剪辑的视频，检测视频中所有人的空间（bounding box）和时间（帧区间）位置，并分类他们正在执行的活动。
        -   这是一个极其具有挑战性的任务，结合了物体检测和动作识别。
        -   示例：框出画面中的人，并标注其动作如“碰杯 -> 饮酒”，“抓取（某人） -> 拥抱”，“看手机 -> 接电话”。

### 二、关键术语定义 (Key Term Definitions)
-   **4D张量 (4D Tensor)**: 用于表示视频数据的四维数组，通常维度顺序为 `时间 x 通道 x 高度 x 宽度` (`T x C x H x W`) 或 `通道 x 时间 x 高度 x 宽度` (`C x T x H x W`)。
-   **视频分类 (Video Classification)**: 一项计算机视觉任务，旨在为整段视频分配一个类别标签，通常这个标签描述的是视频中的主要动作或活动。
-   **单帧CNN (Single-Frame CNN)**: 一种视频分类的基线方法，它独立地处理视频中的每一帧，然后对每帧的预测结果进行平均，以得到整个视频的最终分类。
-   **晚期融合 (Late Fusion)**: 一种视频处理策略，其中模型首先独立地、深入地处理每一帧以提取高层特征，然后在网络的较后阶段才将这些时间上的特征融合起来进行最终决策。
-   **早期融合 (Early Fusion)**: 一种视频处理策略，它在网络的最开始阶段（通常是第一层卷积）就将所有时间帧的信息融合在一起（例如通过堆叠通道），然后使用标准的2D CNN进行后续处理。
-   **3D CNN**: 一种卷积神经网络，使用3D卷积核和3D池化层，能够同时在空间（高和宽）和时间维度上进行卷积操作，从而直接学习时空特征。
-   **慢速融合 (Slow Fusion)**: 3D CNN的一种别称，强调了它在网络的多个层级中逐步、缓慢地融合时空信息的特性，与早期或晚期融合的一次性融合形成对比。
-   **时间平移不变性 (Temporal Shift-Invariance)**: 指模型识别某个动作或模式的能力，不应受到该动作在视频片段中发生早晚的影响。3D CNN通过在时间轴上滑动卷积核来自然地实现这一特性。
-   **光流 (Optical Flow)**: 图像序列中物体运动的二维向量场，表示了图像中每个像素从一帧到下一帧的位移。
-   **双流网络 (Two-Stream Networks)**: 一种视频分类架构，包含两个并行网络：一个处理视频的外观信息（如单帧RGB图像），另一个处理视频的运动信息（如光流），最后融合它们的预测。
-   **循环卷积网络 (Recurrent Convolutional Network, RCN)**: 一种结合了卷积神经网络和循环神经网络思想的模型，其中RNN的矩阵乘法被替换为卷积操作，使得模型可以在处理序列数据的同时保持空间结构。
-   **时空自注意力 (Spatio-Temporal Self-Attention)**: 一种将自注意力机制应用于视频的方法，它允许模型计算视频中所有位置（空间和时间）之间的相互关系，从而捕捉长距离依赖。
-   **非局部块 (Nonlocal Block)**: 实现时空自注意力的一种模块，可以插入到标准的CNN架构中，使其能够捕捉长距离的依赖关系，而不仅仅是局部感受野内的信息。
-   **膨胀3D网络 (Inflated 3D Networks, I3D)**: 一种将预训练的2D CNN架构（如Inception、ResNet）“膨胀”成3D版本的方法，通过复制2D卷积核的权重到新的时间维度并进行缩放，以利用图像领域已学习的特征。
-   **慢快网络 (SlowFast Networks)**: 一种高效的视频识别架构，包含一个处理低帧率、高语义信息（慢路径）的网络，和一个处理高帧率、低语义信息（快路径）的网络，并通过侧向连接融合信息。是当前视频识别领域的先进模型之一。
-   **时序动作定位 (Temporal Action Localization)**: 视频理解任务，旨在从长时间的未剪辑视频中，检测出特定动作的起始和结束时间。
-   **时空检测 (Spatio-Temporal Detection)**: 更复杂的视频理解任务，不仅要检测出视频中动作的发生时间，还要在每一帧中用边界框标出执行该动作的实体（如人），并分类其动作。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **单帧CNN模型 (Single-Frame CNN Model)**:
    1.  输入一个视频片段（T帧）。
    2.  对于 `t = 1 to T` 的每一帧 `frame_t`：
        -   使用一个预训练的2D CNN（如ResNet）计算该帧的分类概率 `prob_t`。
    3.  对所有帧的概率进行平均，得到最终的视频分类概率：`final_prob = (1/T) * Σ(prob_t)`。

-   **晚期融合模型（池化版） (Late Fusion Model with Pooling)**:
    1.  输入一个视频片段（T帧）。
    2.  对于每一帧 `frame_t`，通过一个2D CNN主干网络（去掉最后的分类层）提取其特征图 `feature_map_t`。
    3.  将所有帧的特征图堆叠起来，形成一个4D张量 `T x D x H' x W'`。
    4.  对这个4D张量进行全局时空平均池化 (Average Pool over space and time)，得到一个D维的特征向量。
    5.  将该特征向量送入一个线性分类器，得到最终的分类分数。

-   **早期融合模型 (Early Fusion Model)**:
    1.  输入一个视频片段，其形状为 `T x 3 x H x W`。
    2.  将输入张量重塑 (reshape) 为 `(3T) x H x W`，即将时间信息合并到通道维度。
    3.  将这个重塑后的3D张量作为一个“超多通道图像”送入一个标准的2D CNN网络进行端到端的分类。

-   **3D CNN模型 (3D CNN Model)**:
    1.  输入一个视频片段，其形状为 `3 x T x H x W`。
    2.  通过一系列的3D卷积层和3D池化层进行特征提取。每个3D卷积层都使用3D的卷积核（例如 `kernel_size = (3, 3, 3)`）在时间和空间维度上滑动。
    3.  在网络的最后，使用池化和线性层得到最终的分类分数。

-   **循环卷积网络 (Recurrent Convolutional Network)**:
    1.  **特征提取 (Feature Extraction)**: 使用CNN（2D或3D）从视频的每个时间点（帧或短片段）提取局部特征。
    2.  **序列处理 (Sequence Processing)**: 使用循环神经网络（如LSTM）来处理这些按时间顺序排列的局部特征序列。
        -   **Many-to-one**: RNN 的最终隐藏状态用于视频的整体分类。
        -   **Many-to-many**: RNN 的每个时间步输出一个预测，适用于逐帧分类或视频描述等任务。
    3.  **核心操作**：将传统RNN中向量的矩阵乘法替换为卷积操作，以处理具有空间维度（特征图）的输入。
        -   新的状态 `h_t` 由旧状态 `h_{t-1}` 和当前输入 `x_t` 经过卷积和非线性激活计算得到。
        -   例如，对于香草RNN (`Vanilla RNN`) 的 `h_{t+1} = tanh(W_h h_t + W_x x_t)`，将其中的矩阵乘法 `W_h h_t` 和 `W_x x_t` 替换为2D卷积操作。

-   **膨胀2D网络到3D (Inflating 2D Networks to 3D, I3D) 初始化策略**:
    -   **目标**：利用在图像上预训练的2D CNN权重初始化3D CNN，以加快训练和提高性能。
    -   **步骤**：
        1.  对于2D CNN中的每个 `K_h x K_w` 卷积/池化层。
        2.  将其替换为3D版本 `K_t x K_h x K_w`。
        3.  **初始化权重**：将原始2D卷积核的权重在新的时间维度上复制 `K_t` 次，然后除以 `K_t` (进行平均)。
        4.  这个策略确保了：如果输入视频是“恒定”的（所有帧都相同），那么膨胀后的3D卷积最初会产生与原始2D卷积相同的输出。

-   **视频模型可视化 (Visualizing Video Models)**:
    1.  **前向传播 (Forward Pass)**: 将随机初始化的输入（图像和光流）通过训练好的视频模型，计算特定类别（如“举重”）的分类分数。
    2.  **反向传播 (Backward Pass)**: 计算该分类分数相对于输入图像和光流的梯度。
    3.  **梯度上升 (Gradient Ascent)**: 利用计算出的梯度，迭代地调整输入图像和光流，使其最大化特定类别的分数。
    4.  **正则化 (Regularization)**: 添加额外的项来鼓励光流在空间上平滑，并调整惩罚项以选择“慢速”或“快速”运动的特征。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   (在对比不同融合策略时) 早期融合 (Early Fusion)，晚期融合 (Late Fusion) 和 3D CNN 之间的区别是什么？(What is the difference?)
-   (在讨论 RNN 处理长序列的局限性时) 我们知道如何处理序列！那循环神经网络 (Recurrent Neural Networks) 怎么样？(We know how to handle sequences! How about recurrent networks?)
-   (在介绍 I3D 时) 我们的想法是采用一个2D CNN 架构。那我们能重用图像架构来处理视频吗？(Can we reuse image architectures for video?)
-   (在可视化视频模型时) 你能猜出这是什么动作吗？(Can you guess the action?)
-   (在可视化视频模型时) 这是一个关于“应用眼部化妆”的动作，你猜到了吗？(Can you guess the action? Apply Eye Makeup)

---
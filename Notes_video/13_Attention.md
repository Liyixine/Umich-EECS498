### [📚] 视频学习脚手架: Lecture 13: Attention

### 一、核心内容大纲 (Core Content Outline)
-   **引言 (Introduction)**
    -   回顾循环神经网络 (Recurrent Neural Networks, RNNs) 及其处理序列数据的能力。
    -   RNNs 的应用示例：机器翻译 (Machine Translation) 和图像字幕 (Image Captioning)。
    -   引出注意力机制 (Attention Mechanism) 解决 RNNs 序列处理中的瓶颈问题。
-   **传统序列到序列模型与 RNNs (Traditional Sequence-to-Sequence with RNNs)**
    -   **输入 (Input)**: 序列 $x_1, \dots, x_T$。
    -   **输出 (Output)**: 序列 $y_1, \dots, y_{T'}$。
    -   **编码器 (Encoder)**:
        -   公式: $h_t = f_W(x_t, h_{t-1})$
        -   作用: 将输入序列 (如 "we are eating bread") 编码成一系列隐藏状态 $h_1, \dots, h_T$。
    -   **解码器 (Decoder)**:
        -   公式: $s_t = g_U(y_{t-1}, s_{t-1}, c)$
        -   初始解码器状态 $s_0$ 由编码器的最终隐藏状态 $h_T$ 预测得到。
        -   上下文向量 (Context Vector) $c$ 通常设置为 $h_T$。
        -   解码器在每个时间步接收上一个输出、上一个隐藏状态和固定的上下文向量，生成当前输出。
-   **序列到序列模型的瓶颈问题 (Problem with Sequence-to-Sequence RNNs)**
    -   **问题**: 输入序列的所有信息都被压缩到一个固定大小的上下文向量 (Context Vector) $c$ 中。
    -   对于非常长的序列（如 $T=1000$），单个固定大小的向量很难有效总结所有信息，导致信息丢失或性能下降。
    -   **核心思想**: 在解码器的每个时间步，都使用一个新的上下文向量。
-   **带有注意力机制的序列到序列模型 (Sequence-to-Sequence with RNNs and Attention)**
    -   **编码器 (Encoder)**: 与传统模型相同，生成一系列隐藏状态 $h_1, \dots, h_T$。
    -   **注意力对齐分数计算 (Compute Alignment Scores)**:
        -   公式: $e_{t,i} = f_{att}(s_{t-1}, h_i)$ (其中 $f_{att}$ 是一个多层感知机 MLP)。
        -   $e_{t,i}$ 表示在生成第 $t$ 个输出时，当前解码器隐藏状态 $s_{t-1}$ 与第 $i$ 个编码器隐藏状态 $h_i$ 的相关性。
    -   **注意力权重归一化 (Normalize Attention Weights)**:
        -   公式: $a_{t,i} = \text{softmax}(e_{t,i}, \dots)$
        -   $0 < a_{t,i} < 1$, 且 $\sum a_{t,i} = 1$。
        -   $a_{t,i}$ 表示在生成第 $t$ 个输出时，对第 $i$ 个输入隐藏状态的关注程度。
    -   **上下文向量计算 (Compute Context Vector)**:
        -   公式: $c_t = \sum_i a_{t,i} h_i$
        -   $c_t$ 是编码器所有隐藏状态的加权和，权重由注意力机制动态生成。
    -   **解码器使用动态上下文向量 (Decoder uses Dynamic Context Vector)**:
        -   公式: $s_t = g_U(y_{t-1}, s_{t-1}, c_t)$
        -   解码器在每个时间步使用根据当前输出需求动态生成的 $c_t$。
    -   **可微分性 (Differentiability)**: 整个注意力机制是可微分的，因此可以通过反向传播 (backprop) 进行端到端训练，无需人工标注注意力权重。
    -   **直观理解 (Intuition)**: 上下文向量 $c_t$ 会关注输入序列中与当前输出相关的部分。
-   **带有注意力机制的图像字幕模型 (Image Captioning with RNNs and Attention)**
    -   **CNN 提取图像特征 (CNN for Image Features)**: 使用卷积神经网络 (CNN) 将图像处理成一个特征网格（例如 3x3 的 $h_{i,j}$ 向量）。
    -   **注意力机制关注图像区域 (Attention focuses on Image Regions)**:
        -   注意力机制在生成每个字幕单词时，动态地关注图像特征网格中的相关区域。
        -   计算方式与序列模型类似，只是输入隐藏状态变为图像特征网格中的向量。
    -   **直观理解**: 模型在生成特定单词时，会“看向”图像中最相关的部分。
-   **人类视觉与注意力机制的关联 (Connection to Human Vision: Fovea & Saccades)**
    -   **中央凹 (Fovea)**: 视网膜 (Retina) 中一个微小区域，具有高视觉敏锐度 (high acuity)。
    -   **眼跳 (Saccades)**: 人眼不断快速地移动，使得高清晰度的中央凹能够逐一扫描并处理视觉场景的不同部分，从而构建出清晰的整体视觉。
    -   注意力权重在每个时间步的变化，类似于人眼中的眼跳，使得模型能够动态地“看向”输入数据中最重要的部分。
-   **注意力机制的通用性 (Generalizing the Attention Mechanism)**
    -   注意力机制是一个非常通用的方法，适用于多种将一种数据类型 (X) 转换为另一种数据类型 (Y) 的任务，尤其是当转换过程需要动态关注输入的不同部分时。
    -   **X, Attend, and Y** 模式：
        -   "Show, attend, and tell" (Xu et al, ICML 2015): 图像 (Image) -> 图像区域 (Image Regions) -> 问题 (Question)。
        -   "Ask, attend, and answer" (Xu and Saenko, ECCV 2016): 图像 (Image) + 文本问题 (Text Question) -> 图像区域/文本问题区域 (Image Regions/Text Question Regions) -> 答案 (Answer)。
        -   "Show, ask, attend, and answer" (Kazemi and Elqursh, 2017): 阅读问题文本 (Read text of question), 注意图像区域 (attend to image regions), 产生答案 (produce answer)。
        -   "Listen, attend, and spell" (Chan et al, ICASSP 2016): 处理原始音频 (Process raw audio), 注意音频区域 (attend to audio regions) 同时生成文本 (while producing text)。
        -   "Listen, attend, and walk" (Mei et al, AAAI 2016): 处理文本 (Process text), 注意文本区域 (attend to text regions), 输出导航命令 (output navigation commands)。
        -   "Show, attend, and interact" (Qureshi et al, ICRA 2017): 处理图像 (Process image), 注意图像区域 (attend to image regions), 输出机器人控制命令 (output robot control commands)。
        -   "Show, attend, and read" (Li et al, AAAI 2019): 处理图像 (Process image), 注意图像区域 (attend to image regions), 输出文本 (output text)。
-   **注意力层 (Attention Layer) 的推广 (Generalization)**
    -   **输入 (Inputs)**:
        -   查询向量 (Query vector) $q$: 形状 ($D_Q$)。
        -   输入向量 (Input vectors) $X$: 形状 ($N_X \times D_X$)。
        -   相似性函数 (Similarity function): $f_{att}$。
    -   **计算 (Computation)**:
        -   相似性 (Similarities) $e$: 形状 ($N_X$)。
            -   **变化一 (Change 1)**: 使用点积 (dot product) 计算相似性 $e_i = q \cdot X_i$。
            -   **变化二 (Change 2)**: 使用缩放点积 (scaled dot product) 计算相似性 $e_i = (q \cdot X_i) / \sqrt{D_Q}$。
                -   **原因**: 防止高维向量点积结果过大，导致 softmax 饱和并产生梯度消失问题 (Vanishing Gradients)。
        -   注意力权重 (Attention weights) $a$: 形状 ($N_X$)。
            -   $a = \text{softmax}(e)$
        -   输出向量 (Output vector) $y$: 形状 ($D_X$)。
            -   $y = \sum_i a_i X_i$
    -   **变化三 (Change 3)**: 允许多个查询向量 (Multiple query vectors)。
        -   输入: 查询向量集 $Q$ (Shape: $N_Q \times D_Q$)，输入向量集 $X$ (Shape: $N_X \times D_X$)。
        -   相似性矩阵 $E$ (Shape: $N_Q \times N_X$): $E = QX^T / \sqrt{D_Q}$。
        -   注意力权重矩阵 $A$ (Shape: $N_Q \times N_X$): $A = \text{softmax}(E, \text{dim}=1)$。
        -   输出向量集 $Y$ (Shape: $N_Q \times D_X$): $Y = AX$。
    -   **变化四 (Change 4)**: 分离键 (Key) 和值 (Value) 的表示。
        -   **动机**: 输入 (X) 在注意力机制中扮演双重角色：一是用于计算与查询的相似度（即作为“键”），二是作为加权求和的对象（即作为“值”）。将它们分离可以提供更大的模型灵活性。
        -   输入: 原始输入向量集 $X$ (Shape: $N_X \times D_X$)，可学习的权重矩阵 $W_K$ (键矩阵, Shape: $D_X \times D_Q$)，$W_V$ (值矩阵, Shape: $D_X \times D_V$)，以及查询矩阵 $W_Q$ (Shape: $D_X \times D_Q$)。
        -   键向量 (Key vectors) $K = XW_K$ (Shape: $N_X \times D_Q$)。
        -   值向量 (Value vectors) $V = XW_V$ (Shape: $N_X \times D_V$)。
        -   查询向量 (Query vectors) $Q = XW_Q$ (Shape: $N_X \times D_Q$)。
        -   相似性矩阵 $E = QK^T / \sqrt{D_Q}$ (Shape: $N_Q \times N_X$)。
        -   注意力权重矩阵 $A = \text{softmax}(E, \text{dim}=1)$ (Shape: $N_Q \times N_X$)。
        -   输出向量集 $Y = AV$ (Shape: $N_Q \times D_V$)。
-   **自注意力层 (Self-Attention Layer)**
    -   自注意力是一种特殊形式的注意力层，其中查询 (Query)、键 (Key) 和值 (Value) 都来自于同一个输入序列或集合 $X$。
    -   **置换等变性 (Permutation Equivariant)**: 自注意力层不关心输入向量的顺序，只关心它们之间的相对关系。这意味着自注意力层处理的是向量的“集合 (sets)”，而非“序列 (ordered sequences)”。
    -   **位置编码 (Positional Encoding)**:
        -   **问题**: 由于自注意力层的置换等变性，它无法学习序列的顺序信息。
        -   **解决方案**: 将每个输入向量与一个表示其在序列中位置的编码向量进行拼接 (concatenate)，再送入自注意力层。
-   **掩码自注意力层 (Masked Self-Attention Layer)**
    -   **目的**: 防止模型在生成当前输出时“偷看”序列中未来的信息。
    -   **机制**: 在计算注意力分数 $E$ 后，对于不允许模型关注的未来位置，将其对应的分数设置为负无穷 ($-\infty$)。经过 softmax 后，这些位置的注意力权重将变为零。
    -   **应用**: 主要用于语言模型 (Language Modeling) 任务，例如预测下一个单词 (predict next word)。
-   **多头自注意力层 (Multihead Self-Attention Layer)**
    -   **目的**: 允许模型在不同的“表示子空间 (subspaces of representations)”中并行地关注输入的不同方面。
    -   **机制**: 将输入的查询、键和值向量在维度上分成 $H$ 个“头 (heads)”，每个头独立地执行自注意力计算。然后将所有头的输出拼接 (concatenate) 起来，通常再经过一个线性变换层。
-   **自注意力与卷积神经网络 (CNN) 的结合 (Example: CNN with Self-Attention)**
    -   CNN 处理图像并输出特征图 (Features)。
    -   使用 $1 \times 1$ 卷积层从特征图生成查询 (Queries)、键 (Keys) 和值 (Values) 的网格。
    -   在这些查询、键和值上执行自注意力操作，计算注意力权重矩阵。
    -   将注意力权重与值进行加权求和，得到新的特征表示。
    -   通常会添加一个残差连接 (Residual Connection) 和一个额外的 $1 \times 1$ 卷积。
    -   这个组合形成了一个“自注意力模块 (Self-Attention Module)”。
-   **序列处理的三种方式 (Three Ways of Processing Sequences)**
    1.  **循环神经网络 (Recurrent Neural Network, RNN)**:
        -   适用于有序序列 (Ordered Sequences)。
        -   优点: 擅长处理长序列 (Long Sequences)，最终隐藏状态能够“看到”整个序列。
        -   缺点: 不可并行化 (Not parallelizable)，需要顺序计算隐藏状态。
    2.  **一维卷积 (1D Convolution)**:
        -   适用于多维网格 (Multidimensional Grids)（包括序列）。
        -   优点: 高度并行化 (Highly parallel)，每个输出可以并行计算。
        -   缺点: 不擅长处理非常长的序列（受限于感受野），需要堆叠很多卷积层。
    3.  **自注意力 (Self-Attention)**:
        -   适用于向量的集合 (Sets of Vectors)。
        -   优点: 擅长处理长序列 (每个输出能“看到”所有输入)。
        -   优点: 高度并行化。
        -   缺点: 非常占用内存 (Very memory intensive)。
-   **Transformer 模型 (The Transformer)**
    -   基于论文 "Attention is all you need" (Vaswani et al, NeurIPS 2017)。
    -   **Transformer Block (转换器块)**: 核心构建模块。
        -   输入: 向量集 $x$。
        -   输出: 向量集 $y$。
        -   自注意力是向量之间唯一的交互方式。
        -   层归一化 (Layer Normalization) 和多层感知机 (MLP) 对每个向量独立操作。
        -   高度可伸缩 (Highly scalable)，高度并行化 (Highly parallelizable)。
    -   **Transformer 模型结构**: 是一系列 Transformer Block 的堆叠。
        -   原始论文的编码器 (Encoder) 和解码器 (Decoder) 各有 6 个块。
-   **Transformer 模型的迁移学习 (Transformer: Transfer Learning)**
    -   被誉为“自然语言处理 (Natural Language Processing, NLP) 领域的 ImageNet 时刻”。
    -   **预训练 (Pretraining)**: 从互联网下载大量文本数据。训练一个巨大的 Transformer 模型用于语言建模。
    -   **微调 (Finetuning)**: 在预训练的模型基础上，针对具体的 NLP 任务进行微调。
    -   **模型规模化 (Scaling up Transformers)**:
        -   模型层数、宽度、头数、参数量、数据量和训练时间/硬件都在不断增长。
        -   训练成本极高，例如 Megatron-LM 训练成本约 $430,000 美元。
    -   **关键发现**: Transformer 模型性能似乎随着规模的增大而提升，当前瓶颈是计算能力和内存。
    -   **文本生成示例 (Text Generation Example)**: 演示了 Transformer 模型根据人类输入的提示生成连贯、高质量文本的能力。

### 二、关键术语定义 (Key Term Definitions)
-   **循环神经网络 (Recurrent Neural Network, RNN)**: 一种擅长处理序列数据的神经网络结构，通过循环连接传递信息。
-   **注意力机制 (Attention Mechanism)**: 一种允许模型在处理序列数据时，动态地“关注”输入序列中不同部分的机制。
-   **编码器 (Encoder)**: 序列到序列模型的一部分，将输入序列映射为一系列隐藏状态。
-   **解码器 (Decoder)**: 序列到序列模型的一部分，根据编码器状态和动态上下文生成输出序列。
-   **上下文向量 (Context Vector)**: 在注意力机制中，是编码器所有隐藏状态的加权和，代表当前解码器关注的输入信息摘要。
-   **对齐分数 (Alignment Scores)**: 解码器当前状态与编码器每个隐藏状态之间的相关性或匹配度。
-   **注意力权重 (Attention Weights)**: 对齐分数经过 softmax 归一化后的概率分布，表示模型对输入序列每个部分的关注程度。
-   **中央凹 (Fovea)**: 人类视网膜中视觉最敏锐的区域。
-   **眼跳 (Saccades)**: 人眼快速的跳跃式运动，用于聚焦视觉信息。
-   **缩放点积 (Scaled Dot Product)**: 一种相似性计算方法，通过除以维度平方根来防止 softmax 饱和。
-   **自注意力 (Self-Attention)**: 一种特殊注意力形式，其中查询、键和值都来自同一个输入序列。
-   **置换等变性 (Permutation Equivariant)**: 模型输出的集合与输入集合的顺序无关，只与输入元素本身有关。
-   **位置编码 (Positional Encoding)**: 用于向自注意力模型引入序列顺序信息的技术。
-   **掩码自注意力 (Masked Self-Attention)**: 一种自注意力变体，通过掩码阻止模型访问未来信息，常用于语言建模。
-   **多头自注意力 (Multihead Self-Attention)**: 将自注意力计算并行化到多个“头”中，以捕捉输入的不同表示子空间。
-   **残差连接 (Residual Connection)**: 将层的输入直接加到其输出上的连接，有助于解决梯度消失问题，加速深层网络训练。
-   **Transformer (Transformer)**: 一种完全基于注意力机制的神经网络模型，在自然语言处理领域取得了显著成功。
-   **层归一化 (Layer Normalization)**: 一种归一化技术，独立归一化每个样本的特征，常用于 Transformer。
-   **预训练 (Pretraining)**: 在大量通用数据上训练模型，使其学习通用的特征表示。
-   **微调 (Finetuning)**: 在预训练模型的基础上，针对特定任务使用少量数据进行调整。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **广义注意力层计算 (Generalized Attention Layer Computation)**:
    1.  **查询、键、值生成 (Query, Key, Value Generation)**:
        *   查询向量 (Query vectors) $Q = XW_Q$ (Shape: $N_Q \times D_Q$)
        *   键向量 (Key vectors) $K = XW_K$ (Shape: $N_X \times D_Q$)
        *   值向量 (Value vectors) $V = XW_V$ (Shape: $N_X \times D_V$)
    2.  **相似性计算 (Similarities)**: $E = QK^T / \sqrt{D_Q}$ (Shape: $N_Q \times N_X$)
    3.  **注意力权重归一化 (Attention Weights)**: $A = \text{softmax}(E, \text{dim}=1)$ (Shape: $N_Q \times N_X$)
    4.  **输出向量计算 (Output Vector)**: $Y = AV$ (Shape: $N_Q \times D_V$)

-   **Transformer Block 结构**:
    *   Input: Set of vectors x (输入向量集 x)
    *   Output: Set of vectors y (输出向量集 y)
    *   **核心**: 自注意力 (Self-Attention) 是向量之间唯一的交互方式。
    *   **组成**:
        1.  自注意力层 (Self-Attention Layer)
        2.  残差连接 (Residual Connection)
        3.  层归一化 (Layer Normalization)
        4.  多层感知机 (MLP) (独立作用于每个向量)
        5.  残差连接 (Residual Connection)
        6.  层归一化 (Layer Normalization)
    *   **特点**: 高度可伸缩 (Highly scalable)，高度并行化 (Highly parallelizable)。

本次视频未包含具体的代码实现细节，仅展示了算法的数学公式和概念图。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   你如何知道这些模型在处理序列时，是按照顺序来理解的？
-   在生成“estamos”（我们是）时，模型可能应该对英文输入“we”和“are”分配较高的注意力权重，对其他词分配较低的权重，对吗？
-   在生成“comiendo”（吃）时，模型可能应该对英文输入“eating”分配较高的注意力权重，对吗？
-   如果模型是置换等变的，那么我们如何让它知道序列中元素的顺序信息？
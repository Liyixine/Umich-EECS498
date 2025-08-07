### [📚] 视频学习脚手架: Lecture 22: Course Recap Open Problems in Computer Vision

### 一、核心内容大纲 (Core Content Outline)

-   **课程回顾与展望 (Course Recap and Outlook)**
    -   本学期课程主题: 深度学习在计算机视觉中的应用 (Deep Learning for Computer Vision)
    -   计算机视觉的定义 (Definition of Computer Vision): 构建能够处理、感知并推理视觉数据的人工系统。
    -   核心挑战 (Core Challenges):
        -   语义鸿沟 (Semantic Gap): 人类对图像的理解与计算机处理的像素数值之间的差异。
        -   视觉数据复杂性 (Complexity of Visual Data): 视角 (Viewpoint)、光照 (Illumination)、形变 (Deformation)、遮挡 (Occlusion)、杂乱 (Clutter)、类内变异 (Intraclass Variation) 等问题。
-   **机器学习：数据驱动方法 (Machine Learning: Data-Driven Approach)**
    -   基本范式 (Basic Paradigm):
        1.  收集图像和标签数据集 (Collect a dataset of images and labels)。
        2.  使用机器学习训练分类器 (Use Machine Learning to train a classifier)。
        3.  在新图像上评估分类器 (Evaluate the classifier on new images)。
    -   深度卷积网络 (Deep Convolutional Networks) 的兴起 (Rise of Deep Convolutional Networks)
        -   ImageNet大规模视觉识别挑战赛 (ImageNet Large Scale Visual Recognition Challenge): 1000个目标类别，1,431,167张图像。
        -   深度学习的进入 (Enter Deep Learning, 2012): 错误率显著下降，超越传统方法。
        -   深度学习的爆发式增长 (Deep Learning Explosion, 2012-Present): CVPR (计算机视觉与模式识别国际会议) 论文提交量和接受量呈指数级增长。
-   **深度学习发展历程 (History of Deep Learning)**
    -   不是一夜之间发明的 (Wasn't invented overnight!):
        -   早期神经元模型 (Early Neuron Models): Perceptron (Frank Rosenblatt, ~1957)、Simple and Complex cells (Hubel & Wiesel, 1959)。
        -   新认知机 (Neocognitron, Fukushima, 1980): 卷积网络早期形式。
        -   反向传播 (Backpropagation, 1985)。
        -   LeNet (LeCun et al, 1998): 现代卷积网络雏形。
        -   AI 冬天 (AI Winter)
        -   AlexNet (Krizhevsky, Sutskever, and Hinton, 2012): 结合大数据、GPU算力及ReLU、动量等优化，实现ImageNet突破。
        -   2018年图灵奖 (2018 Turing Award): 授予Yoshua Bengio、Geoffrey Hinton、Yann LeCun 以表彰他们在深度学习领域的贡献。
-   **神经网络训练核心要素 (Core Elements of Neural Network Training)**
    -   简单分类器 (Simple Classifiers):
        -   k-近邻分类器 (k-Nearest Neighbor classifier)。
        -   线性分类器 (Linear Classifiers): $y = Wx + b$。
    -   优化 (Optimization):
        -   梯度下降 (Gradient Descent) 及随机梯度下降 (Stochastic Gradient Descent, SGD)。
        -   梯度下降的问题 (Problems with Gradient Descent): 局部最小值 (Local Minima)、鞍点 (Saddle Points)、病态条件 (Poor Conditioning)、梯度噪声 (Gradient Noise)。
        -   梯度下降改进 (Gradient Descent Improvements): 动量 (Momentum)、Nesterov、AdaGrad、RMSProp、Adam 等自适应学习率优化器 (Adaptive Learning Rates)。
    -   更复杂的模型：神经网络 (More Complex Models: Neural Networks)
        -   全连接层 (Fully-Connected Layers)。
        -   激活函数 (Activation Function): Sigmoid、tanh、ReLU、Leaky ReLU、Maxout、ELU。
        -   万能近似定理 (Universal Approximation): 单隐藏层神经网络可近似任何连续函数。
        -   卷积神经网络 (Convolutional Networks): 卷积层 (Convolution Layers)、池化层 (Pooling Layers)、归一化层 (Normalization, e.g., Batch Normalization)。
        -   经典卷积网络架构 (Classical CNN Architectures): AlexNet、VGG16/19、GoogLeNet、ResNet。
        -   效率关注 (Efficiency): ResNeXt、MobileNets (使用分组卷积等)。
    -   网络表示：计算图 (Representing Networks: Computational Graphs): 将神经网络表示为计算图，便于反向传播。
    -   计算梯度：反向传播 (Computing Gradients: Backpropagation): 通过链式法则高效计算梯度。
    -   深度学习硬件与软件 (Deep Learning Hardware and Software): CPU、GPU、TPU。PyTorch (动态图) 与 TensorFlow (静态图)。
    -   训练技巧 (Training Tips):
        -   数据预处理 (Data Preprocessing): 零均值化 (zero-centered)、归一化 (normalized)。
        -   权重初始化 (Weight Initialization): 确保激活值在各层中保持良好的缩放。
        -   数据增强 (Data Augmentation): 通过变换图像来增加数据量和引入不变性。
        -   正则化 (Regularization): 增加随机性以避免过拟合 (Batch Normalization, Dropout, Cutout, Stochastic Depth, Mixup, Fractional Pooling)。
        -   学习率调度 (Learning Rate Schedules): 调整学习率以优化训练过程。
        -   选择超参数 (Choosing Hyperparameters): 网格搜索 (Grid Layout) 与随机搜索 (Random Layout)，观察学习曲线 (Learning Curves)。
-   **计算机视觉任务与前沿进展 (Computer Vision Tasks and Frontiers)**
    -   常见任务 (Common Tasks): 图像分类 (Classification)、语义分割 (Semantic Segmentation)、目标检测 (Object Detection)、实例分割 (Instance Segmentation)。
    -   循环神经网络 (Recurrent Neural Networks): 用于序列数据处理 (Process Sequences)，如图像标注 (Image Captioning)。
        -   架构 (Architectures): Vanilla Recurrent Network, Long Short Term Memory (LSTM)。
        -   图像标注 (Image Captioning): 结合CNN提取特征和RNN生成文本。
        -   注意力机制 (Attention): 让模型在处理序列时关注图像的不同区域。
    -   自注意力层 (Self-Attention Layer) 与 Transformer (The Transformer): 基于自注意力机制的通用架构，在自然语言处理和视觉任务中表现出色。
    -   三维深度学习 (3D Deep Learning):
        -   三维CNN (3D CNNs) 处理视频数据。
        -   三维形状表示 (3D Shape Representations): 深度图 (Depth Map)、体素网格 (Voxel Grid)、隐式表面 (Implicit Surface)、点云 (Pointcloud)、网格 (Mesh)。
        -   Mesh R-CNN。
    -   视频深度学习 (Deep Learning on Video):
        -   三维CNN (3D CNNs): 卷积操作同时覆盖空间和时间维度。
        -   双流网络 (Two Stream Networks): 分别处理空间信息和时间信息（光流）。
        -   自注意力机制在视频处理中的应用。
    -   生成模型 (Generative Models):
        -   自回归模型 (Autoregressive Models): 直接最大化训练数据的似然。
        -   变分自编码器 (Variational Autoencoders, VAEs): 引入潜在变量 Z，最大化证据下界 (Evidence Lower Bound, ELBO)。
        -   生成对抗网络 (Generative Adversarial Networks, GANs): 不显式建模概率分布，通过对抗学习生成样本。
    -   强化学习 (Reinforcement Learning):
        -   智能体 (Agents) 与环境 (Environment) 交互，学习最大化奖励 (maximize reward)。
        -   Q-学习 (Q-Learning): 训练 Q 函数估计未来奖励。
        -   策略梯度 (Policy Gradients): 训练策略网络直接输出动作分布。
-   **未来展望 (What's Next?)**
    -   预测 #1 (Prediction #1): 将发现有趣的全新深度模型类型 (Discover interesting new types of deep models)。
        -   例如：神经常微分方程 (Neural Ordinary Differential Equations, Neural ODE): 将残差网络视为连续时间上的微分方程解，模型深度趋于无限。
    -   预测 #2 (Prediction #2): 深度学习将找到新的应用 (Find new applications)。
        -   科学应用 (Scientific applications): 医学影像 (Medical Imaging) 诊断、星系分类 (Galaxy Classification)、鲸鱼识别 (Whale Recognition) 等。
        -   将基本监督学习思想应用于更多现实世界问题。
        -   深度学习应用于计算机科学 (Deep Learning for Computer Science): 改进传统数据结构，如哈希表 (Hash Table)。
        -   深度学习应用于数学 (Deep Learning for Mathematics): 定理证明 (Theorem Proving)、符号积分 (Symbolic Integration)。
    -   预测 #3 (Prediction #3): 深度学习将使用更多数据和计算 (Deep Learning will use more data and compute)。
        -   计算成本 (Cost of Computation): GigaFLOPs per Dollar (吉浮点每美元) 呈指数级下降，得益于GPU和TPU发展。
        -   AI训练所需的计算量 (Compute used for AI Training): Petaflop/s-days (千万亿次浮点运算每天) 自1950年代以来呈超指数增长。
        -   新硬件 (New Hardware): 如Cerebras Wafer Scale Engine (晶圆级引擎)，提供比最大GPU大几个数量级的计算能力，专门用于深度学习。
-   **深度学习面临的问题 (Problems with Deep Learning)**
    -   问题 #1 (Problem #1): 模型存在偏见 (Models are biased)。
        -   **性别偏见 (Gender Bias)**: 词向量 (Word Vectors) 在学习语言规律时会习得社会中存在的性别刻板印象。
        -   **经济偏见 (Economic Bias)**: 视觉分类器 (Visual Classifiers) 在识别物体时对高收入西方家庭的物体识别效果更好，对其他地区和收入水平的物体识别效果差。
        -   **种族偏见 (Racial Bias)**: 2015年Google相册曾错误地将非洲裔美国人标记为“大猩猩”。
        -   解决方向：建立更公平、无偏见的机器学习模型 (Making ML Work for Everyone)。
    -   问题 #2 (Problem #2): 需要新理论? (Need new theory?)
        -   **经验之谜：好的子网络 (Empirical Mystery: Good Subnetworks)**:
            -   彩票假设 (Lottery Ticket Hypothesis): 在随机初始化的深度网络中，存在一个在训练后剪枝 (pruning) 掉大部分权重后，用原始初始化值训练也能达到与完整网络相似性能的子网络。
            -   未训练的子网络 (Untrained Subnet): 甚至可以在随机初始化且未经训练的网络中找到能进行分类的子网络。
            -   暗示我们对深度网络训练和初始化过程存在根本性缺失的理解。
        -   **经验之谜：泛化 (Empirical Mystery: Generalization)**:
            -   经典统计学习理论 (Classical Statistical Learning Theory) 预测：模型复杂度增加会导致训练误差降低，但测试误差在某点后会上升（过拟合）。
            -   深度网络能完美拟合随机标签数据而不会在真实数据上过拟合，这与经典理论相悖。
            -   双降现象 (Double Descent): 经验观察到，当模型复杂度超过某个阈值后，测试误差在达到峰值后会再次下降。
            -   意味着需要新的理论来解释深度学习的泛化能力。
    -   问题 #3 (Problem #3): 深度学习需要大量标注训练数据 (Deep Learning needs a lot of labeled training data)。
        -   标注数据成本高昂。
        -   **少样本学习 (Low-Shot Learning) 的新数据集**:
            -   Omniglot Dataset (Omniglot数据集): 大量类别但每类样本极少，挑战模型从少量样本中学习。
            -   KMNIST Dataset (KMNIST数据集): 类似MNIST，但字符更复杂，样本数量不均。
            -   LVIS Dataset (LVIS数据集): 大规模实例分割数据集，包含长尾分布的类别，挑战少样本识别。
        -   **使用未标注数据：自监督学习 (Using Unlabeled Data: Self-Supervised Learning)**:
            -   **步骤**:
                1.  在无需标注数据的“预训练任务 (pretext task)”上训练CNN。
                2.  在目标任务上对CNN进行微调 (fine-tune)，希望只需少量标注数据。
            -   **预训练任务示例**:
                -   解决拼图 (Jigsaw Puzzles): 学习图像的空间结构和部分之间的关系。
                -   图像上色 (Colorization): 将灰度图转为彩色图，学习图像的语义和颜色关联。
                -   图像修补 (Inpainting): 填充图像中的缺失区域，学习图像的上下文信息。
            -   **自监督学习的最新进展 (State of the Art)**: 包括对比预测编码 (Contrastive Predictive Coding)、对比多视图编码 (Contrastive Multiview Coding)、动量对比 (Momentum Contrast)、预训练不变性表示 (Pretext-Invariant Representations) 等。
    -   问题 #4 (Problem #4): 深度学习不“理解”世界 (Deep Learning doesn't "Understand" the world)。
        -   模型学习的是对数据的模仿，而非对世界的真正理解。
        -   **语言模型缺乏常识 (Language Models lack common sense)**:
            -   GPT-2等大型语言模型在处理涉及常识推理的问题时，会给出语法正确但逻辑荒谬的回答。
        -   **目标检测器很脆弱 (Object Detectors are brittle)**:
            -   在训练数据以外的稍微不同的图像上，模型表现可能灾难性地下降。
            -   例如，在房间里加入一只PS的大象，检测器可能将大象错认为椅子，甚至影响对其他物体的正确识别。
            -   这表明CNNs以与人类截然不同的方式“看”世界，它们不能泛化到训练数据中未见的上下文。

-   **总结 (Conclusion)**
    -   **预测 (Predictions)**: 将有新的深度学习模型、新的应用、更多的计算和新硬件。
    -   **问题 (Problems)**: 模型存在偏见、需要新理论、依赖大量数据、缺乏对世界的理解。
    -   现在是在计算机视觉和机器学习领域工作的绝佳时机！

### 二、关键术语定义 (Key Term Definitions)

-   **神经常微分方程 (Neural Ordinary Differential Equations, Neural ODE)**: 一种新型的深度学习模型，将神经网络的层视为连续的函数，通过求解常微分方程来表示隐藏状态的演变，模型深度可以趋于无限。
-   **数值积分 (Numerical Integration)**: 近似计算定积分的方法，在神经常微分方程中，残差网络层之间的更新步骤被类比为数值积分的离散步骤。
-   **哈希冲突 (Hash Collisions)**: 在哈希表中，不同的键经过哈希函数计算后得到相同的哈希值，导致它们映射到同一个存储位置的现象。
-   **图神经网络 (Graph Neural Networks)**: 一类设计用于直接在图结构数据上运行的神经网络，能够学习节点和边之间的复杂关系。
-   **定理证明 (Theorem Proving)**: 计算机科学和数学逻辑中的一个领域，旨在使用计算机程序自动或半自动地证明数学定理。
-   **符号积分 (Symbolic Integration)**: 计算机代数系统中的一个任务，用于找到给定函数的原函数（不定积分），结果是另一个函数表达式，而不是数值。
-   **吉浮点每美元 (GigaFLOPs per Dollar)**: 一种衡量计算成本效益的指标，表示每美元可以获得的十亿次浮点运算能力。
-   **千万亿次浮点运算每天 (Petaflop/s-days)**: 衡量AI模型训练所需计算量的单位，表示以每秒千万亿次浮点运算的速度运行一天所需的计算量。
-   **晶圆级引擎 (Wafer Scale Engine, WSE)**: 一种超大规模的集成电路芯片，尺寸接近整个硅晶圆，旨在提供极高的计算能力和内存带宽，通常用于深度学习加速。
-   **偏见 (Bias)**: 在机器学习模型中，由于训练数据中固有的不公平或不准确的模式，导致模型对特定群体或情况做出不公平或不准确的预测或决策。
-   **词向量 (Word Vectors)**: 在自然语言处理中，将词语映射到连续向量空间中的低维实数向量，捕获词语的语义和句法关系。
-   **少样本学习 (Low-Shot Learning)**: 机器学习的一个子领域，旨在使模型能够在只有少量（甚至一个）训练样本的情况下，快速学习和识别新的类别或任务。
-   **预训练任务 (Pretext Task)**: 在自监督学习中，用于在大量无标注数据上预训练模型的一个辅助任务，通常是设计来迫使模型学习有用的数据表示。
-   **微调 (Fine-tune)**: 在深度学习中，指在一个已在大型数据集或相关任务上预训练过的模型的基础上，使用较小的数据集或特定任务对模型进行进一步训练，以适应新的任务。
-   **自监督学习 (Self-Supervised Learning)**: 一种机器学习范式，模型利用数据本身的结构或属性来生成监督信号，从而在没有人类标注的情况下进行学习。
-   **拼图 (Jigsaw Puzzles)**: 在自监督学习中，一种预训练任务，模型需要将打乱的图像块重新组合成原始图像，从而学习图像的空间关系和特征。
-   **上色 (Colorization)**: 在自监督学习中，一种预训练任务，模型需要将灰度图像转换为彩色图像，从而学习图像中的语义信息和颜色分布。
-   **图像修补 (Inpainting)**: 在自监督学习中，一种预训练任务，模型需要填充图像中缺失的区域，从而学习图像的上下文和结构信息。
-   **双降现象 (Double Descent)**: 经验观察到的机器学习模型性能现象，即随着模型复杂度（或数据量）的增加，测试误差在传统过拟合区域达到峰值后，会再次下降。
-   **插值阈值 (Interpolation Threshold)**: 双降现象中，训练误差达到零（模型完全拟合训练数据）的复杂度点。
-   **彩票假设 (Lottery Ticket Hypothesis)**: 提出在随机初始化的神经网络中，包含着一些特殊的“子网络”（即“彩票”），它们在独立训练时能比随机初始化的完整网络达到更好的性能。
-   **常识 (Common Sense)**: 人类普遍具备的、不需正式推理就能理解和应用的关于世界的基本知识和直觉。
-   **大象在房间里 (The Elephant in the Room)**: 比喻一个显而易见但被刻意回避或不被讨论的问题。在计算机视觉中，指的是模型在面对不常见但显而易见的物体时，会表现出理解能力的缺失。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **机器学习数据驱动方法步骤 (Machine Learning Data-Driven Approach Steps)**:
    1.  收集数据集 (Collect a dataset of images and labels)。
    2.  使用机器学习训练分类器 (Use Machine Learning to train a classifier)。
    3.  在新图像上评估分类器 (Evaluate the classifier on new images)。

-   **训练和预测函数示例 (Example Training and Prediction Functions)**:
    ```python
    def train(images, labels):
        # 机器学习!
        return model

    def predict(model, test_images):
        # 使用模型预测标签
        return test_labels
    ```

-   **梯度下降算法骨架 (Gradient Descent Algorithm Skeleton)**:
    ```python
    # Vanilla 梯度下降
    w = initialize_weights()
    for t in range(num_steps):
        dw = compute_gradient(loss_fn, data, w)
        w -= learning_rate * dw
    ```

-   **线性分类器公式 (Linear Classifier Formula)**:
    $y = Wx + b$

-   **残差网络 (Residual Network) 层更新公式**:
    $h_{t+1} = h_t + f(h_t, \theta_t)$

-   **神经常微分方程 (Neural ODE) 隐藏状态演变公式**:
    $dh/dt = f(h(t), t, \theta)$

### 四、讲师提出的思考题 (Questions Posed by the Instructor)

本次视频未明确提出思考问题。

---
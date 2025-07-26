### [📚] 视频学习脚手架: Lecture 10: Training Neural Networks (Part 1)

### 一、核心内容大纲 (Core Content Outline)
-   **前情回顾: 硬件与软件 (Last Time: Hardware and Software)**
    -   回顾 CPU, GPU, TPU 等不同硬件类型及其应用 (Reviewed different hardware types like CPU, GPU, and TPU and their applications).
    -   讨论静态图与动态图的差异 (Discussed the differences between static and dynamic graphs).
    -   比较 PyTorch 和 TensorFlow 等软件系统及其优劣 (Compared software systems like PyTorch and TensorFlow and their trade-offs).
-   **训练神经网络总览 (Overview of Training Neural Networks)**
    -   本次课程及后续课程将涵盖的三个主要阶段 (Three main phases covered in this lecture and the next):
        1.  **一次性设置 (One-time Setup)**: 训练过程开始前的准备工作 (Preparation before the training process starts).
            -   激活函数 (Activation Functions)
            -   数据预处理 (Data Preprocessing)
            -   权重初始化 (Weight Initialization)
            -   正则化 (Regularization)
        2.  **训练动态 (Training Dynamics)**: 优化过程中的调整 (Adjustments during the optimization process).
            -   学习率调度 (Learning Rate Schedules)
            -   大批量训练 (Large-batch Training)
            -   超参数优化 (Hyperparameter Optimization)
        3.  **训练后处理 (After Training)**: 模型训练完成后的额外步骤 (Additional steps after model training).
            -   模型集成 (Model Ensembles)
            -   迁移学习 (Transfer Learning)
-   **激活函数 (Activation Functions)**
    -   **必要性**: 非线性激活函数对神经网络的处理能力至关重要，防止多层线性操作退化为单层线性操作 (Non-linear activation functions are critical for neural network processing power, preventing multi-layer linear operations from collapsing into a single linear layer).
    -   **Sigmoid 激活函数 (Sigmoid Activation Function)**: $\sigma(x) = 1 / (1 + e^{-x})$
        -   将数值压缩到 (Squashes numbers to range).
        -   历史上流行，可以解释为神经元的饱和“发放率”或概率 (Historically popular as it can be interpreted as a saturating "firing rate" of a neuron or a probability).
        -   **3 个主要问题 (3 Main Problems)**:
            1.  **饱和神经元“杀死”梯度 (Saturated neurons "kill" the gradients)**:
                -   当输入 `x` 过小或过大时，局部梯度接近于零，导致权重更新非常缓慢或停止 (When input `x` is very small or very large, the local gradient is very close to zero, leading to very slow or no weight updates).
                -   这在深度网络中会导致梯度消失问题 (This leads to the vanishing gradient problem in deep networks).
            2.  **输出非零均值 (Outputs are not zero-centered)**:
                -   所有 Sigmoid 输出都是正数 (All Sigmoid outputs are positive).
                -   导致权重梯度始终同号，引起训练过程中的锯齿形优化路径，降低收敛速度 (Causes weight gradients to always have the same sign, leading to zig-zagging optimization paths and slower convergence).
            3.  `exp()` **计算成本高 (exp() is computationally expensive)**:
                -   指数函数计算复杂，在 CPU 和移动设备上效率较低 (Exponential function is computationally complex, less efficient on CPUs and mobile devices).
    -   **Tanh 激活函数 (Tanh Activation Function)**: `tanh(x)`
        -   将数值压缩到 [-1, 1] 范围 (Squashes numbers to range [-1, 1]).
        -   **零均值 (Zero-centered)**: 解决了 Sigmoid 的输出非零均值问题 (Solves Sigmoid's non-zero-centered output problem).
        -   **仍存在梯度饱和问题 (Still kills gradients when saturated)**: 梯度消失问题依旧存在 (Vanishing gradient problem persists).
    -   **ReLU (修正线性单元) 激活函数 (ReLU (Rectified Linear Unit) Activation Function)**: `f(x) = max(0, x)`
        -   在正值区域不饱和 (Does not saturate in the positive region).
        -   计算效率高，只需简单阈值判断 (Very computationally efficient, only a simple threshold check).
        -   实践中收敛速度比 Sigmoid/Tanh 快得多 (e.g., 6倍) (Converges much faster than Sigmoid/Tanh in practice, e.g., 6x).
        -   **输出非零均值 (Not zero-centered output)**:
        -   **“死亡 ReLU”问题 ("Dying ReLU" Problem)**:
            -   当输入 `x < 0` 时，梯度正好为零，导致神经元停止学习 (When input `x < 0`, the gradient is exactly zero, causing the neuron to stop learning).
            -   一旦 ReLU 神经元进入此状态，其权重将不再更新 (Once a ReLU neuron enters this state, its weights will never update).
            -   解决方案: 有时将 ReLU 神经元的偏差初始化为略正的值 (e.g., 0.01) (Sometimes initialize ReLU neurons with slightly positive biases (e.g., 0.01)).
    -   **Leaky ReLU 激活函数 (Leaky ReLU Activation Function)**: `f(x) = max(0.01x, x)`
        -   在负值区域有小的正斜率，因此不会饱和且不会“死亡” (Has a small positive slope in the negative region, so it doesn't saturate or "die").
        -   计算效率高 (Computationally efficient).
        -   在负值区域的斜率 (e.g., 0.01) 是需要调优的超参数 (The slope (e.g., 0.01) in the negative region is a hyperparameter to tune).
    -   **Parametric Rectifier (PReLU) 激活函数 (Parametric Rectifier (PReLU) Activation Function)**: `f(x) = max(\alpha x, x)`
        -   类似 Leaky ReLU，但负值区域的斜率 `\alpha` 是网络可学习的参数 (Similar to Leaky ReLU, but the slope `\alpha` in the negative region is a learnable parameter of the network).
    -   **Exponential Linear Unit (ELU) 激活函数 (Exponential Linear Unit (ELU) Activation Function)**: `f(x) = x` (if `x > 0`), `f(x) = \alpha (exp(x) - 1)` (if `x <= 0`)
        -   具有 ReLU 的所有优点 (All benefits of ReLU).
        -   输出更接近零均值 (Closer to zero mean outputs).
        -   负饱和区域增加了对噪声的鲁棒性 (Negative saturation regime adds robustness to noise).
        -   计算需要 `exp()` 函数，因此计算成本较高 (Computation requires `exp()` function, thus computationally expensive).
    -   **Scaled Exponential Linear Unit (SELU) 激活函数 (Scaled Exponential Linear Unit (SELU) Activation Function)**:
        -   ELU 的缩放版本，在深度网络中表现更好 (Scaled version of ELU that works better for deep networks).
        -   具有“自归一化”特性，可以在不使用 Batch Normalization 的情况下训练深度网络 (Has a "Self-Normalizing" property; can train deep SELU networks without Batch Normalization).
        -   数学推导复杂 (Derivation is complex).
    -   **激活函数性能总结 (Activation Function Performance Summary)**
        -   CIFAR10 上的准确率比较显示，大多数现代激活函数在性能上差异不大 (Accuracy comparison on CIFAR10 shows most modern activation functions have similar performance).
        -   在实践中，不同激活函数之间的性能差异通常在 1% 以内，并且趋势不一致 (In practice, performance differences are usually within 1% and trends are not consistent).
        -   **建议 (Advice)**:
            -   不要过度思考，直接使用 **ReLU** (Don't think too hard. Just use ReLU). [27:39]
            -   如果需要“压榨”最后 0.1% 的性能，可以尝试 Leaky ReLU / ELU / SELU / GELU (Try out Leaky ReLU / ELU / SELU / GELU if you need to squeeze that last 0.1%). [27:48]
            -   **不要使用 Sigmoid 或 Tanh**，它们通常会导致网络难以收敛 (Don't use sigmoid or tanh, they often prevent networks from converging). [28:18]
-   **数据预处理 (Data Preprocessing)**
    -   **目的**: 使数据更适合高效的神经网络训练 (Purpose: Make data more amenable to efficient neural network training). [30:57]
    -   **常见技术 (Common Techniques)**:
        1.  **零均值化 (Zero-centering)**:
            -   从每个特征中减去训练集的均值，使数据中心位于原点 (Subtract the mean of the training data from each feature, centering the data at the origin). [31:56]
            -   代码示例 (Code Example): `X -= np.mean(X, axis = 0)` [31:56]
            -   **重要性**: 避免所有权重梯度始终同号的问题 (Importance: Avoids the problem of weight gradients always having the same sign). [32:19]
        2.  **归一化 (Normalization)**:
            -   将每个特征除以其在训练集上的标准差，使每个特征具有单位方差 (Divide each feature by its standard deviation on the training set, giving each feature unit variance). [32:00]
            -   代码示例 (Code Example): `X /= np.std(X, axis = 0)` [32:00]
        -   **效果**: 预处理后的数据更集中，损失函数对权重变化不再敏感，更容易优化 (Effect: Preprocessed data is more concentrated, making the loss function less sensitive to weight changes and easier to optimize). [34:17]
    -   **高级技术 (Advanced Techniques)** (非图像数据中更常见) (More common in non-image data):
        -   **PCA (主成分分析) (Principal Component Analysis)**: 对数据进行去相关，旋转数据云使其特征与坐标轴对齐 (Decorrelates data, rotating the data cloud so features align with coordinate axes). [33:09]
        -   **白化 (Whitening)**: 在去相关数据后进一步缩放，使每个特征具有单位方差，协方差矩阵变为单位矩阵 (Further scales decorrelated data so each feature has unit variance, making the covariance matrix an identity matrix). [33:28]
        -   **图像数据预处理 (Data Preprocessing for Images)**:
            -   **减去均值图像 (Subtract the mean image)** (例如 AlexNet): 计算训练集所有图像的均值图像 (32x32x3 数组)，然后从每个图像中减去 (Compute the mean image of all training images (32x32x3 array), then subtract it from each image). [36:51]
            -   **减去每通道均值 (Subtract per-channel mean)** (例如 VGGNet): 计算每个颜色通道（R, G, B）的均值，然后从对应通道的像素值中减去 (Compute the mean for each color channel (R, G, B), then subtract it from the pixel values of the corresponding channel). [37:13]
            -   **减去每通道均值并除以每通道标准差 (Subtract per-channel mean and divide by per-channel std)** (例如 ResNet): 这是最常见的图像预处理方法 (This is the most common image preprocessing method). [37:29]
            -   **注意事项**: 始终在训练集上计算统计数据，并应用于训练和测试集，以模拟实际部署场景 (Always compute statistics on the training set and apply them to both training and test sets to simulate real-world deployment). [37:57]
            -   **不常见**: 对图像数据进行 PCA 或白化通常不常见 (PCA or whitening are not common for image data). [36:40]
-   **权重初始化 (Weight Initialization)**
    -   **问题**: 如果所有权重都初始化为零或常数，所有神经元将学习相同的东西，导致对称性问题和梯度为零，无法学习 (Problem: If all weights are initialized to zero or constants, all neurons learn the same thing, leading to symmetry issues and zero gradients, preventing learning). [39:17, 40:01]
    -   **解决方案: 小随机数 (Small Random Numbers)**:
        -   从零均值的高斯分布中采样小随机数 (e.g., std=0.01) 进行初始化 (Initialize with small random numbers sampled from a Gaussian distribution with zero mean (e.g., std=0.01)). [40:19]
        -   对小型网络有效，但对深度网络有缺陷 (Works okay for small networks but has problems with deeper networks). [40:40]
        -   **激活统计 (Activation Statistics)**:
            -   对于深层网络 (例如 6 层 tanh 网络，隐藏层大小 4096)，如果权重过小，激活值会趋向于零，导致梯度消失 (For deep networks (e.g., 6-layer tanh network, hidden size 4096), if weights are too small, activations tend to zero, leading to vanishing gradients). [41:20]
            -   如果权重过大 (例如 std=0.05)，激活值会饱和，梯度也会趋向于零 (If weights are too large (e.g., std=0.05), activations saturate, also leading to zero gradients). [42:51]
    -   **Xavier 初始化 (Xavier Initialization)**:
        -   通过设置标准差为 `1 / sqrt(Din)` 来解决激活值过小或过大的问题，其中 `Din` 是输入维度 (Solves the issue of too small/large activations by setting std to `1 / sqrt(Din)`, where `Din` is the input dimension). [43:51]
        -   **推导**: 目标是使输出的方差等于输入的方差，以保持激活值分布的稳定性 (Derivation: Goal is for variance of output to equal variance of input, maintaining stable activation distribution). [44:45]
        -   对于全连接层，`Din` 是输入神经元的数量 (For fully-connected layers, `Din` is the number of input neurons).
        -   对于卷积层，`Din` 是 `kernel_size * kernel_size * input_channels` (For conv layers, `Din` is `kernel_size * kernel_size * input_channels`). [44:21]
        -   **问题**: Xavier 假定激活函数是零均值的（例如 Tanh），对 ReLU 等非零均值激活函数效果不好，会导致激活值仍趋于零 (Problem: Xavier assumes zero-centered activation functions (like Tanh), but performs poorly with non-zero-centered ones like ReLU, causing activations to collapse to zero). [47:33]
    -   **Kaiming / MSRA 初始化 (Kaiming / MSRA Initialization)**:
        -   为 ReLU 激活函数设计的初始化方法，标准差为 `sqrt(2 / Din)` (Initialization method designed for ReLU activation functions, std is `sqrt(2 / Din)`). [48:04]
        -   解决了 ReLU 的“死亡”问题，使激活值在深层网络中仍能保持良好尺度 (Solves the "dying ReLU" problem, keeping activations well-scaled in deep networks). [48:17]
    -   **残差网络初始化 (Weight Initialization for Residual Networks)**:
        -   对于残差块内部的第一个卷积层，使用 MSRA 初始化 (For the first convolution layer within a residual block, use MSRA initialization). [50:25]
        -   对于残差块内部的第二个卷积层，将其权重初始化为零 (For the second convolution layer within a residual block, initialize its weights to zero). [50:29]
        -   这样，在初始化时，残差块近似于一个恒等映射，避免了方差爆炸问题 (This way, at initialization, the residual block approximates an identity function, preventing variance explosion). [50:39]
    -   **总结**: 适当的初始化是一个活跃的研究领域 (Proper initialization is an active area of research). [50:46]
-   **正则化 (Regularization)**
    -   **目的**: 防止模型过拟合训练数据 (Purpose: Prevent the model from overfitting the training data). [52:00]
    -   **常见模式 (Common Pattern)**:
        -   **训练时**: 添加某种随机性 (Training: Add some kind of randomness).
        -   **测试时**: 对随机性进行平均（有时近似） (Testing: Average out the randomness (sometimes approximate)).
    -   **L2 正则化 (L2 Regularization)** (也称为权重衰减 Weight Decay):
        -   在损失函数中添加权重平方和的惩罚项 (Adds a penalty term of the sum of squared weights to the loss function). [52:33]
        -   强制权重变小，从而使模型更简单，减少过拟合 (Forces weights to be smaller, making the model simpler and reducing overfitting).
    -   **L1 正则化 (L1 Regularization)**:
        -   在损失函数中添加权重绝对值和的惩罚项 (Adds a penalty term of the sum of absolute values of weights to the loss function). [52:45]
        -   倾向于使不重要的特征的权重变为零，实现特征选择 (Tends to drive weights of unimportant features to zero, enabling feature selection).
    -   **Elastic Net (L1 + L2)**:
        -   结合 L1 和 L2 正则化 (Combines L1 and L2 regularization). [52:49]
    -   **Dropout (随机失活)**:
        -   **训练时**: 在每个前向传播中，随机将部分神经元（及其连接）的输出设置为零。丢弃的概率 `p` 是超参数，0.5 常见 (Training: Randomly set outputs of some neurons (and their connections) to zero in each forward pass. Dropping probability `p` is a hyperparameter, 0.5 is common). [53:05]
        -   **效果**: 强制网络学习冗余表示，防止特征之间的“协同适应”（即神经元过度依赖特定输入） (Effect: Forces the network to have a redundant representation; prevents co-adaptation of features). [53:52]
        -   **测试时**: 所有神经元都保持激活，但其输出按 `p` 缩放，以近似平均训练时的随机性 (Testing: All neurons are active, but their outputs are scaled by `p` to approximate averaging out randomness from training). [58:41]
        -   **“倒置 Dropout (Inverted Dropout)”**: 更常见的实现，训练时直接将保留的神经元输出乘以 `1/p`，测试时无需缩放 (More common implementation: Scale retained neuron outputs by `1/p` during training; no scaling needed during testing). [59:52]
        -   **应用**: 通常应用于网络末端的全连接层，这些层参数量大，容易过拟合 (Typically applied to large fully-connected layers at the end of the network, which are prone to overfitting). [60:58]
    -   **Batch Normalization (批量归一化)**:
        -   **训练时**: 对每个 Batch 的特征进行归一化，使用当前 Batch 的均值和方差 (Training: Normalize features within each batch using the mean and variance of that batch). [62:38]
        -   **测试时**: 使用训练集所有 Batch 的固定运行均值和方差进行归一化 (Testing: Use fixed running mean and variance accumulated from all training batches for normalization). [63:08]
        -   **作为正则化器**: Batch Normalization 本身具有正则化效果，因为每个样本的归一化都依赖于当前 Batch 的其他样本，引入了随机性 (Batch Normalization itself acts as a regularizer because normalization of each sample depends on other samples in the batch, introducing randomness). [62:45]
        -   **影响**: ResNet 等较新的架构通常只使用 L2 正则化和 Batch Normalization，而不再使用 Dropout (Newer architectures like ResNet often only use L2 regularization and Batch Normalization, and no longer use Dropout). [63:18]
    -   **数据增强 (Data Augmentation)**:
        -   **原理**: 对原始训练数据进行保留标签的随机变换，人为地增加训练集的大小和多样性 (Principle: Apply label-preserving random transformations to the original training data, artificially increasing the size and diversity of the training set). [64:25]
        -   **效果**: 提升模型的泛化能力，防止过拟合 (Effect: Improves model generalization and prevents overfitting).
        -   **常见图像增强方法 (Common Image Augmentation Methods)**:
            -   水平翻转 (Horizontal flips) [64:41]
            -   随机裁剪和缩放 (Random crops and scales) [65:28]
            -   色彩抖动 (Color jittering): 随机调整图像的对比度和亮度，或更复杂地沿主成分方向添加颜色偏移 (Randomly adjust contrast and brightness, or more complex color offsets along principal component directions). [66:44]
            -   其他 (Others): 平移 (translation), 旋转 (rotation), 拉伸 (stretching), 剪切 (shearing), 镜头畸变 (lens distortions) 等。可以根据具体问题发挥创意 (translation, rotation, stretching, shearing, lens distortions, etc. Be creative for your problem). [66:51]
        -   **测试时**: 通常使用原始图像进行评估，或对固定集合的增强图像进行平均预测 (Testing: Usually evaluate on original images, or average predictions over a fixed set of augmented images). [65:58]
    -   **DropConnect**:
        -   **训练时**: 随机将神经元之间的连接（权重）设置为零 (Training: Randomly set connections (weights) between neurons to zero). [68:08]
        -   **测试时**: 使用所有连接 (Testing: Use all connections).
    -   **Fractional Max Pooling (分形最大池化)**:
        -   **训练时**: 使用随机化的池化区域 (Training: Use randomized pooling regions). [68:24]
        -   **测试时**: 平均不同样本的预测 (Testing: Average predictions over different samples).
    -   **Stochastic Depth (随机深度)**:
        -   **训练时**: 随机跳过 ResNet 中的一些残差块 (Training: Randomly skip some residual blocks in ResNet). [69:01]
        -   **测试时**: 使用整个网络 (Testing: Use the whole network).
    -   **Cutout (剪裁)**:
        -   **训练时**: 将图像中的随机区域设置为零 (Training: Set random image regions to zero). [69:31]
        -   **测试时**: 使用整个图像 (Testing: Use the whole image).
        -   **适用性**: 对 CIFAR 等小型数据集效果很好，对 ImageNet 等大型数据集不太常见 (Works very well for small datasets like CIFAR, less common for large datasets like ImageNet).
    -   **Mixup (混合)**:
        -   **训练时**: 随机混合一对图像的像素值，并按比例混合其标签 (Training: Randomly blend pixel values of pairs of images and proportionally blend their labels). [69:58]
        -   **测试时**: 使用原始图像 (Testing: Use original images).
        -   **效果**: 强制模型在训练时探索样本之间的平滑过渡，提高泛化能力 (Effect: Forces the model to explore smooth transitions between samples during training, improving generalization).

### 二、关键术语定义 (Key Term Definitions)
-   **静态图 (Static Graph)**: 在执行前完全定义计算图，编译后执行，通常用于 TensorFlow。 (A computational graph defined entirely before execution, then compiled and run, typically seen in TensorFlow.)
-   **动态图 (Dynamic Graph)**: 计算图在运行时动态构建，更灵活，常用于 PyTorch。 (A computational graph built dynamically at runtime, offering more flexibility, commonly used in PyTorch.)
-   **激活函数 (Activation Function)**: 神经网络中引入非线性的函数，将神经元的加权输入转换为输出。 (A function in a neural network that introduces non-linearity, transforming the weighted input of a neuron into its output.)
-   **Sigmoid 激活函数 (Sigmoid Activation Function)**: 一种 S 形的激活函数，将输入压缩到 (Squashes numbers to range).
-   **Tanh 激活函数 (Tanh Activation Function)**: 双曲正切激活函数，将输入压缩到 [-1, 1] 范围，且是零均值的。 (Hyperbolic tangent activation function that squashes inputs to the range [-1, 1] and is zero-centered.)
-   **ReLU (修正线性单元) (ReLU (Rectified Linear Unit))**: 一种激活函数，输出为 `max(0, x)`，即负输入为零，正输入保持不变。 (An activation function that outputs `max(0, x)`, meaning negative inputs become zero while positive inputs remain unchanged.)
-   **死亡 ReLU (Dying ReLU)**: ReLU 神经元在训练过程中停止激活并停止更新权重的现象。 (A phenomenon where a ReLU neuron stops activating and its weights stop updating during training.)
-   **Leaky ReLU 激活函数 (Leaky ReLU Activation Function)**: ReLU 的变体，在负值区域有一个小的非零斜率 (e.g., 0.01x)，以避免“死亡 ReLU”问题。 (A variant of ReLU that has a small, non-zero slope (e.g., 0.01x) for negative inputs, to prevent the "dying ReLU" problem.)
-   **Parametric Rectifier (PReLU) 激活函数 (Parametric Rectifier (PReLU) Activation Function)**: Leaky ReLU 的变体，负值区域的斜率是可学习的参数。 (A variant of Leaky ReLU where the slope in the negative region is a learnable parameter.)
-   **ELU (指数线性单元) (ELU (Exponential Linear Unit))**: 一种激活函数，在负值区域使用指数函数，并具有零均值输出的倾向。 (An activation function that uses an exponential function for negative inputs and tends to produce zero-mean outputs.)
-   **SELU (缩放指数线性单元) (SELU (Scaled Exponential Linear Unit))**: ELU 的缩放版本，旨在使神经网络在深度增加时保持激活值的均值和方差不变，从而实现“自归一化”。 (A scaled version of ELU designed to maintain constant mean and variance of activations as network depth increases, enabling "self-normalization".)
-   **数据预处理 (Data Preprocessing)**: 在将数据输入神经网络之前对其进行转换，以提高训练效率和模型性能。 (Transforming data before feeding it into a neural network to improve training efficiency and model performance.)
-   **零均值化 (Zero-centering)**: 数据预处理的一种方法，通过从每个特征中减去均值来使数据的均值为零。 (A data preprocessing technique that involves subtracting the mean from each feature to make the data have a mean of zero.)
-   **归一化 (Normalization)**: 数据预处理的一种方法，通过除以标准差来缩放数据，使每个特征具有单位方差。 (A data preprocessing technique that involves scaling data by dividing by the standard deviation, so each feature has unit variance.)
-   **PCA (主成分分析) (Principal Component Analysis)**: 一种降维和去相关技术，通过旋转数据将其特征对齐到主成分方向。 (A dimensionality reduction and decorrelation technique that rotates data to align its features with principal components.)
-   **白化 (Whitening)**: 数据预处理的一种方法，它去除了特征之间的相关性，并使每个特征具有相同的方差（通常为1）。 (A data preprocessing technique that removes correlations between features and makes each feature have the same variance (usually 1).)
-   **权重初始化 (Weight Initialization)**: 在神经网络训练开始时，为网络中的权重参数设置初始值。 (Setting initial values for the weight parameters in a neural network before training begins.)
-   **Xavier 初始化 (Xavier Initialization)**: 一种权重初始化方法，旨在保持各层激活值的方差稳定，避免梯度消失或爆炸。适用于零均值激活函数（如 Tanh）。 (A weight initialization method aiming to keep the variance of activations stable across layers, preventing vanishing or exploding gradients. Suitable for zero-mean activation functions like Tanh.)
-   **Kaiming / MSRA 初始化 (Kaiming / MSRA Initialization)**: 一种权重初始化方法，为 ReLU 激活函数设计，通过考虑 ReLU 的非线性特性来保持激活值方差的稳定性。 (A weight initialization method designed for ReLU activation functions, maintaining stable activation variance by accounting for ReLU's non-linear property.)
-   **正则化 (Regularization)**: 旨在防止模型过拟合训练数据，提高泛化能力的策略。 (Strategies aimed at preventing a model from overfitting the training data and improving its generalization ability.)
-   **L2 正则化 (L2 Regularization)**: 一种正则化技术，通过在损失函数中添加权重平方和的惩罚项来减小模型权重。 (A regularization technique that penalizes the sum of the squares of the weights in the loss function to reduce model weights.)
-   **L1 正则化 (L1 Regularization)**: 一种正则化技术，通过在损失函数中添加权重绝对值和的惩罚项来鼓励稀疏模型（即许多权重为零）。 (A regularization technique that penalizes the sum of the absolute values of the weights in the loss function to encourage sparse models (i.e., many weights are zero).)
-   **Elastic Net**: L1 和 L2 正则化的结合。 (A combination of L1 and L2 regularization.)
-   **Dropout (随机失活)**: 一种正则化技术，在训练过程中随机地将部分神经元（及其连接）的输出设置为零。 (A regularization technique where randomly selected neurons (and their connections) are temporarily dropped out of the network during training.)
-   **DropConnect**: 类似于 Dropout，但随机断开的是神经元之间的连接（权重），而不是整个神经元。 (Similar to Dropout, but instead of dropping entire neurons, random connections (weights) between neurons are dropped.)
-   **数据增强 (Data Augmentation)**: 通过对现有训练数据进行随机变换（如翻转、裁剪、缩放、颜色抖动等）来扩充数据集，提高模型的泛化能力。 (Expanding the dataset by applying random transformations (e.g., flips, crops, scaling, color jitter) to existing training data, to improve model generalization.)
-   **分形最大池化 (Fractional Max Pooling)**: 一种池化技术，使用随机化的、非整数比例的池化区域，引入随机性。 (A pooling technique that uses randomized, non-integer ratios for pooling regions, introducing randomness.)
-   **随机深度 (Stochastic Depth)**: 一种正则化技术，在训练时随机跳过深度网络（如 ResNet）中的一些残差块。 (A regularization technique that randomly skips some residual blocks in deep networks (like ResNet) during training.)
-   **Cutout (剪裁)**: 一种数据增强技术，在训练图像中随机遮挡一个正方形区域，以强制模型关注图像的更多部分。 (A data augmentation technique that involves randomly masking out a square region in training images, to force the model to focus on more parts of the image.)
-   **Mixup (混合)**: 一种数据增强技术，通过线性插值组合两个训练样本及其对应的标签，生成新的训练样本。 (A data augmentation technique that generates new training samples by linearly interpolating two training samples and their corresponding labels.)

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **数据零均值化 (Zero-centering Data)**:
    ```python
    X -= np.mean(X, axis = 0)
    ```

-   **数据归一化 (Normalizing Data)**:
    ```python
    X /= np.std(X, axis = 0)
    ```

-   **权重初始化: 小随机数 (Weight Initialization: Small Random Numbers)**:
    ```python
    W = 0.01 * np.random.randn(Din, Dout)
    ```

-   **权重初始化: Xavier 初始化 (Weight Initialization: Xavier Initialization)**:
    ```python
    W = np.random.randn(Din, Dout) / np.sqrt(Din)
    ```

-   **权重初始化: Kaiming / MSRA 初始化 (Weight Initialization: Kaiming / MSRA Initialization)**:
    ```python
    W = np.random.randn(Din, Dout) * np.sqrt(2 / Din) # ReLU correction
    ```

-   **正则化: Dropout 前向传播 (Regularization: Dropout Forward Pass)**:
    ```python
    # p = 0.5 # probability of keeping a unit active. higher = less dropout
    def train_step(X):
        """ X contains the data """
        # forward pass for example 3-layer neural network
        H1 = np.maximum(0, np.dot(W1, X) + b1)
        U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
        H1 *= U1 # drop!

        H2 = np.maximum(0, np.dot(W2, H1) + b2)
        U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
        H2 *= U2 # drop!

        out = np.dot(W3, H2) + b3

        # backward pass: compute gradients... (not shown)
        # perform parameter update... (not shown)
    ```

-   **正则化: Dropout 测试时缩放 (Regularization: Dropout Scaling at Test Time)**:
    ```python
    def predict(X):
        # ensembled forward pass
        H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
        H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
        out = np.dot(W3, H2) + b3
    ```

-   **正则化: Inverted Dropout (倒置 Dropout)** (更常见，测试时无需缩放) (More common, no scaling needed at test time):
    ```python
    # p = 0.5 # probability of keeping a unit active. higher = less dropout
    def train_step(X):
        """ X contains the data """
        # forward pass for example 3-layer neural network
        H1 = np.maximum(0, np.dot(W1, X) + b1)
        U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
        H1 *= U1 # drop!

        H2 = np.maximum(0, np.dot(W2, H1) + b2)
        U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
        H2 *= U2 # drop!

        out = np.dot(W3, H2) + b3

        # backward pass: compute gradients... (not shown)
        # perform parameter update... (not shown)

    def predict(X):
        # ensembled forward pass
        H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
        H2 = np.maximum(0, np.dot(W2, H1) + b2) # no scaling necessary
        out = np.dot(W3, H2) + b3
    ```

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   对于 Sigmoid 激活函数，当 `x = -10` 时会发生什么？当 `x = 0` 时会发生什么？当 `x = 10` 时会发生什么？ (For the Sigmoid activation function, what happens when `x = -10`? What happens when `x = 0`? What happens when `x = 10`?)
-   当神经元的输入始终为正时，关于权重 `w` 的梯度我们能说什么？ (When the input to a neuron is always positive, what can we say about the gradients on `w`?)
-   当 `x < 0` 时，ReLU 的梯度是多少？ (What is the gradient of ReLU when `x < 0`?)
-   如果我们把所有的 `W` 都初始化为 0，`b` 也初始化为 0，会发生什么？ (What happens if we initialize all W=0, b=0?)
-   Xavier 初始化法中，如果所有激活函数都趋于零，梯度会是怎样的？ (For Xavier Initialization, if all activations tend to zero, what do the gradients dL/dW look like?)
-   Xavier 初始化法中，如果所有激活函数都饱和了，梯度会是怎样的？ (For Xavier Initialization, if all activations saturate, what do the gradients dL/dW look like?)

---

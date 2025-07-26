### [📚] 视频学习脚手架: 神经网络 (Neural Networks)

### 一、核心内容大纲 (Core Content Outline)
-   **讲座介绍与回顾 (Lecture Introduction and Review)**
    -   讲座主题：神经网络 (Neural Networks) - 深度学习模型 (Deep Learning Models) 的核心。
    -   前期课程回顾：
        -   使用线性模型 (Linear Models) 解决图像分类 (Image Classification) 问题。
        -   使用损失函数 (Loss Functions) 量化模型对权重 (Weights) 选择的满意度（如 Softmax, SVM）。
        -   使用随机梯度下降 (Stochastic Gradient Descent, SGD) 及其变种（如 Momentum, Adam, RMSProp）最小化损失函数并训练模型。
    -   本讲座重点：从线性模型转向更强大的神经网络模型。
-   **线性分类器 (Linear Classifiers) 的局限性 (Limitations)**
    -   几何视角 (Geometric Viewpoint):
        -   线性分类器通过高维超平面 (High-dimensional Hyperplanes) 将空间划分为两部分。
        -   无法有效分离非线性可分 (Non-linearly Separable) 的数据点（例如：同心圆分布）。
    -   视觉视角 (Visual Viewpoint):
        -   每个类别仅学习一个模板 (One Template per Class)。
        -   无法识别同一类别中不同模式 (Different Modes)（例如：马头朝左或朝右）。
-   **特征转换 (Feature Transforms) 解决方案 (Solutions)**
    -   核心思想：将原始数据 (Original Space) 转换到新的特征空间 (Feature Space)，使其更易于分类。
    -   示例：将笛卡尔坐标 (Cartesian Coordinates) (x, y) 转换为极坐标 (Polar Coordinates) (r, θ)。
        -   在线性不可分的原空间，通过特征转换，数据在新的特征空间中变得线性可分 (Linearly Separable)。
    -   图像特征 (Image Features)：
        -   **颜色直方图 (Color Histogram)**: 统计图像颜色分布。
        -   **方向梯度直方图 (Histogram of Oriented Gradients, HoG)**: 描述图像纹理和形状。
        -   **视觉词袋 (Bag of Words, BoW)**: 数据驱动学习特征，通过聚类图像块形成“视觉词汇”。
    -   特征融合 (Feature Fusion): 组合多种特征表示以创建更强大的特征向量。
-   **神经网络结构与实现 (Neural Network Architecture and Implementation)**
    -   **2层神经网络 (2-layer Neural Network)**: $ f = W_2 \max(0, W_1 x) $
        -   可以将其理解为：第一层学习一组模板 (Bank of Templates)，第二层对这些模板进行重新组合 (Recombines Templates)。
        -   可以识别同一类别中多种模式（例如：马头朝左或朝右）。
        -   **分布式表示 (Distributed Representation)**：学习到的模板通常不具备人类可解释性。
    -   **深度神经网络 (Deep Neural Networks)**: 通过堆叠更多层 (Layers) 来增加深度 (Depth)。
        -   深度 (Depth) = 层数（权重矩阵的数量）。
        -   宽度 (Width) = 每层（隐藏层）的维度大小。
        -   **全连接神经网络 (Fully-Connected Neural Network)** 或 **多层感知机 (Multi-Layer Perceptron, MLP)**：每层的每个神经元都与下一层的所有神经元连接。
    -   **激活函数 (Activation Functions)**: 引入非线性 (Non-linearity) 的关键。
        -   如果神经网络没有激活函数，它仍然是一个线性分类器。
        -   常见的激活函数类型：Sigmoid, tanh, ReLU, Leaky ReLU, Maxout, ELU。
        -   **ReLU (Rectified Linear Unit)**: $ \max(0, x) $ 是目前深度学习中默认的最佳选择 (Good Default Choice)。
-   **神经网络与生物神经元 (Neural Networks and Biological Neurons)**
    -   **生物神经元 (Biological Neurons) 功能**: 树突 (Dendrite) 接收信号，细胞体 (Cell Body) 整合，轴突 (Axon) 传导信号，突触 (Synapse) 连接。发放率 (Firing Rate) 是输入信号的非线性函数。
    -   **人工神经元 (Artificial Neuron) 模型**: 对加权输入求和，然后应用激活函数输出。
    -   **对比与注意 (Comparison and Caution)**:
        -   相似性：人工神经元是对生物神经元的一种粗略抽象。
        -   **差异与警告 (Differences and Warnings)**：
            -   生物神经元有多种类型，树突能执行复杂非线性计算，突触是复杂的动态系统，发放率编码可能不足够。
            -   生物神经元连接模式复杂、不规则，甚至有循环 (Loops)。而传统人工神经网络通常是分层 (Layered) 且前馈 (Feedforward) 的，这是为了计算效率而非生物真实性。
            -   近年研究显示，具有随机连接模式 (Random Connectivity Patterns) 的人工神经网络也能工作。
            -   因此，应**非常小心地使用大脑类比 (Brain Analogies)**，"Neural"一词更多是历史遗留。
-   **空间扭曲 (Space Warping) - 神经网络的几何解释 (Geometric Interpretation of Neural Networks)**
    -   **线性变换 (Linear Transform)**: $ h = Wx $。它对输入空间进行线性变形，但无法将非线性可分的数据点变为线性可分。
    -   **非线性空间扭曲 (Non-linear Space Warping) 与 ReLU**: $ h = \text{ReLU}(Wx) $
        -   ReLU 激活函数会“折叠 (folding)”或“压缩 (collapsing)”输入空间中的区域。
        -   例如，输入空间中的负值区域会被 ReLU 压缩到 0。
        -   通过这种非线性变换，原本在原始空间中非线性可分的数据点，在新的特征空间中可以变得线性可分 (Linearly Separable)。
        -   在特征空间中训练一个线性分类器，对应回原始空间就是一个非线性分类器。
        -   更多的隐藏单元 (More Hidden Units) 意味着更多的容量 (More Capacity)，可以学习到更复杂的决策边界 (More Complex Decision Boundaries)。
-   **正则化 (Regularization) 与模型容量 (Model Capacity)**
    -   过多的模型容量可能导致过拟合 (Overfitting)。
    -   **不推荐通过减少模型大小 (Don't regularize with size)** 来进行正则化。
    -   **推荐使用更强的 L2 正则化 (Instead, use stronger L2 regularization)**。L2 正则化可以通过惩罚大的权重来平滑决策边界，从而提高模型的泛化能力。
-   **通用近似定理 (Universal Approximation Theorem)**
    -   **陈述 (Statement)**: 一个带有一个隐藏层 (One Hidden Layer) 的神经网络，可以以任意精度 (Arbitrary Precision) 近似任何连续函数 (Any Continuous Function) $f: R^N \to R^M$。
    -   **直观理解 (Intuitive Understanding)**:
        -   单个 ReLU 单元产生一个具有平坦区域和线性区域的函数。
        -   通过组合多个 ReLU 单元（例如四个隐藏单元），可以构建一个“凹凸函数 (bump function)”。
        -   通过叠加 K 个这样的“凹凸函数”，可以以任意精度近似任何复杂的连续函数。
    -   **通用近似定理告诉我们 (Universal Approximation tells us)**: 神经网络具有强大的表示能力 (Can Represent Any Function)。
    -   **通用近似定理没有告诉我们 (Universal Approximation DOES NOT tell us)**:
        -   我们是否真的能用随机梯度下降 (SGD) 来学习 (Learn) 任何函数。
        -   我们需要多少数据 (How Much Data) 来学习一个函数。
    -   **重要提醒 (Important Note)**: kNN (K-Nearest Neighbors) 也是一个通用近似器 (Universal Approximator)。因此，仅仅具有通用近似能力并不意味着模型是“最好”的。
-   **凸函数 (Convex Functions) 与优化 (Optimization)**
    -   **定义 (Definition)**: 一个函数 $f: X \subseteq R^N \to R$ 是凸函数，如果对于所有 $x_1, x_2 \in X, t \in$，满足 $f(tx_1 + (1-t)x_2) \le tf(x_1) + (1-t)f(x_2)$。
    -   **几何直观 (Geometric Intuition)**: 凸函数是一个（多维的）碗形 (Multidimensional Bowl)。连接函数上任意两点的割线 (Secant Line) 总是位于函数上方或与之重合。
    -   **性质 (Properties)**:
        -   通常容易优化 (Easy to Optimize)。
        -   可以推导关于收敛到全局最小值 (Global Minimum) 的理论保证。
        -   局部最小值 (Local Minima) 也是全局最小值。
    -   **线性分类器 (Linear Classifiers) 的优化**: 损失函数是凸函数，因此优化问题是凸优化问题，易于求解。
-   **非凸优化 (Non-convex Optimization)**
    -   **神经网络的优化**: 大多数神经网络的损失函数是非凸的 (Non-convex)，这使得优化变得困难。
    -   **挑战 (Challenges)**:
        -   很少或没有关于收敛的理论保证 (Few or No Guarantees about Convergence)。
        -   损失函数表面可能非常复杂，有许多局部最小值。
    -   **经验观察 (Empirical Observation)**: 尽管缺乏理论保证，但在实践中，非凸优化算法（如 SGD）似乎仍然有效。
    -   这是一个活跃的研究领域 (Active Area of Research)。
-   **下一步 (Next Steps)**
    -   将学习如何计算神经网络中的梯度 (Gradients)。
    -   下一讲：反向传播 (Backpropagation) 算法。

### 二、关键术语定义 (Key Term Definitions)
-   **线性模型 (Linear Models)**: 在机器学习中，通过线性函数对输入数据进行建模，用于分类或回归问题。
-   **损失函数 (Loss Functions)**: 量化模型预测值与真实值之间差异的函数，用于评估模型的性能。
-   **随机梯度下降 (Stochastic Gradient Descent, SGD)**: 一种优化算法，用于最小化损失函数，通过在每次迭代中随机选择一个样本或小批量样本来计算梯度并更新模型参数。
-   **特征转换 (Feature Transforms)**: 将原始输入数据映射到新的特征空间，以期在新空间中数据更易于被模型处理和学习。
-   **颜色直方图 (Color Histogram)**: 图像特征之一，通过统计图像中不同颜色值（或颜色范围）的像素数量来表示图像的颜色分布。
-   **方向梯度直方图 (Histogram of Oriented Gradients, HoG)**: 图像特征之一，通过计算图像局部区域内边缘方向的分布来描述图像的纹理和形状信息。
-   **视觉词袋 (Bag of Words, BoW)**: 一种图像表示方法，通过从图像中提取的局部图像块进行聚类形成“视觉词汇”，然后用这些词汇在图像中的出现频率来表示图像。
-   **ImageNet 挑战赛 (ImageNet Challenge)**: 一个大规模的图像识别竞赛，推动了计算机视觉和深度学习领域的发展。
-   **多层感知机 (Multi-Layer Perceptron, MLP)**: 一种包含至少一个隐藏层 (Hidden Layer) 的前馈神经网络 (Feedforward Neural Network)，其中层与层之间是全连接 (Fully-Connected) 的。
-   **激活函数 (Activation Functions)**: 神经网络中引入非线性 (Non-linearity) 的函数，应用于神经元的加权输入之和，决定神经元的输出。
-   **修正线性单元 (Rectified Linear Unit, ReLU)**: 一种常用的激活函数，其定义为 $ f(x) = \max(0, x) $，即当输入大于 0 时输出输入值，否则输出 0。
-   **生物神经元 (Biological Neuron)**: 大脑的基本组成单位，负责接收、处理和传递电化学信号。
-   **树突 (Dendrite)**: 神经元的接收部分，接收来自其他神经元的电脉冲。
-   **轴突 (Axon)**: 神经元的传导部分，将电脉冲从细胞体传导出去。
-   **突触 (Synapse)**: 神经元之间传递信号的连接点，是轴突末梢与另一个神经元树突或细胞体之间的微小间隙。
-   **细胞体 (Cell Body)**: 神经元的核心部分，包含细胞核，并整合来自树突的信号。
-   **发放率 (Firing Rate)**: 神经元在单位时间内产生动作电位（电脉冲）的频率，常用于描述神经元的活动水平。
-   **空间扭曲 (Space Warping)**: 神经网络通过非线性激活函数对输入空间进行的变换，使其在新的特征空间中数据更容易被分类。
-   **通用近似定理 (Universal Approximation Theorem)**: 指出具有至少一个隐藏层和非线性激活函数的神经网络可以近似任何连续函数。
-   **凸函数 (Convex Functions)**: 具有特定数学性质的函数，其上任意两点连线（割线）总位于函数图上方或与之重合，在优化中易于找到全局最小值。
-   **非凸优化 (Non-convex Optimization)**: 对非凸函数进行优化，通常存在多个局部最小值，导致优化过程复杂且难以保证收敛到全局最小值。
-   **反向传播 (Backpropagation)**: 神经网络中用于计算损失函数梯度 (Gradient) 的核心算法，是训练深度学习模型的关键。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **2层神经网络的数学定义 (Mathematical Definition of 2-layer Neural Network)**:
    -   $ f(x; W_1, W_2) = W_2 \max(0, W_1 x) $
    -   其中 $ x \in R^D $ 是输入向量，$ W_1 \in R^{H \times D} $ 是第一层权重矩阵，$ H $ 是隐藏层维度，$ W_2 \in R^{C \times H} $ 是第二层权重矩阵，$ C $ 是类别数。
    -   $ \max(0, \cdot) $ 是 ReLU 激活函数。

-   **深度神经网络的通用表示 (General Representation of Deep Neural Networks)**:
    -   对于一个具有 $ L $ 个隐藏层的深度神经网络，其输出 $ s $ 可以表示为：
    -   $ s = W_L \max(0, W_{L-1} \max(0, \dots \max(0, W_1 x))) $
    -   其中 $ W_k $ 是第 $ k $ 层的权重矩阵。

-   **简单神经网络的 NumPy 实现 (Simple Neural Network Implementation in NumPy)**:
    ```python
    import numpy as np
    from numpy.random import randn

    # N: batch size
    # Din: input dimension
    # H: hidden dimension
    # Dout: output dimension
    N, Din, H, Dout = 64, 1000, 100, 10

    # Initialize weights and data
    x = randn(N, Din)  # Input data
    y = randn(N, Dout) # True output (for training)

    w1 = randn(Din, H) # Weights for the first layer
    w2 = randn(H, Dout) # Weights for the second layer

    # Training loop
    for t in range(10000):
        # Forward pass: compute predicted y
        # Sigmoid activation function for hidden layer
        h = 1.0 / (1.0 + np.exp(-x.dot(w1)))
        y_pred = h.dot(w2)

        # Compute loss
        loss = np.square(y_pred - y).sum()
        # print(t, loss) # Uncomment to see loss decreasing

        # Backpropagation: compute gradients
        dy_pred = 2.0 * (y_pred - y)
        dw2 = h.T.dot(dy_pred) # Gradient for w2

        # Sigmoid derivative for hidden layer gradient
        dh = dy_pred.dot(w2.T)
        dw1 = x.T.dot(dh * h * (1 - h)) # Gradient for w1

        # SGD step: update weights
        learning_rate = 1e-4
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
    ```

-   **Universal Approximation Theorem 的数学形式和示例 (Mathematical Form and Example of Universal Approximation Theorem)**:
    -   对于一个输入为 $x$ (单维)，输出为 $y$ (单维)，且包含一个具有 3 个隐藏单元的隐藏层，并使用 ReLU 激活函数的 2 层神经网络：
    -   隐藏单元的激活值:
        $h_1 = \max(0, w_1 x + b_1)$
        $h_2 = \max(0, w_2 x + b_2)$
        $h_3 = \max(0, w_3 x + b_3)$
    -   输出值:
        $y = u_1 h_1 + u_2 h_2 + u_3 h_3 + p$
    -   代入隐藏单元的表达式:
        $y = u_1 \max(0, w_1 x + b_1) + u_2 \max(0, w_2 x + b_2) + u_3 \max(0, w_3 x + b_3) + p$
    -   这表示输出是多个“移位 (shifted)”、“缩放 (scaled)”的 ReLU 函数之和。
    -   通过巧妙地组合和设置权重，可以构建一个“凹凸函数 (bump function)”。
    -   具有 4K 隐藏单元的神经网络可以构建 K 个凹凸函数之和，从而近似任何连续函数。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   如果构建一个没有激活函数的神经网络，会发生什么？(What happens if we build a neural network with no activation function?)
-   (在解释激活函数时，有同学提问) max 函数的目的是什么？(What was the purpose of the max?)
-   (在解释模板视觉化时) 为什么这些学习到的模板会出现重复结构？(Why do these learned templates seem to have repeated structures?)
-   (在通用近似定理部分) 这种方法有什么缺点？(What about... Gaps between bumps? Other nonlinearities? Higher-dimensional functions?)
-   (在通用近似定理部分) 通用近似定理告诉了我们什么，没有告诉我们什么？(Universal approximation tells us: Neural nets can represent any function. Universal approximation DOES NOT tell us: Whether we can actually learn any function with SGD, How much data we need to learn a function.)

---
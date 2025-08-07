### [📚] 视频学习脚手架: Generative Models Part 1 - Lecture 19

### 一、核心内容大纲 (Core Content Outline)
-   **讲座介绍与回顾 (Lecture Introduction & Recap)**
    -   欢迎来到第 19 讲：生成模型，第一部分 (Welcome to Lecture 19: Generative Models, Part 1) [0:00]
    -   上次回顾：视频模型 (Last Time: Videos) [0:11]
        -   讨论了多种视频处理模型，如单帧 CNN (Single-frame CNN)、后期融合 (Late Fusion)、早期融合 (Early Fusion)、3D CNN / C3D [0:37]
        -   以及双流网络 (Two-stream Networks)、CNN + RNN、卷积 RNN (Convolutional RNN)、时空自注意力 (Spatio-temporal Self-attention) 和 SlowFast 网络 (SlowFast Networks) [0:40]
-   **生成模型简介 (Introduction to Generative Models)** [0:58]
    -   本讲将重点探讨生成模型 (Generative Models) [1:06]
    -   预告：本讲内容将涉及更多数学和更少图示 [1:15]
    -   本讲将涵盖变分自编码器 (Variational Autoencoders) 和自回归模型 (Autoregressive Models)，下一讲将讨论生成对抗网络 (Generative Adversarial Networks) [1:45]
-   **监督学习 vs. 无监督学习 (Supervised vs. Unsupervised Learning)** [2:06]
    -   **监督学习 (Supervised Learning)**
        -   数据形式: $(x, y)$，其中 $x$ 为数据， $y$ 为标签 (Data: x, y; x is data, y is label) [2:18]
        -   目标: 学习一个将 $x$ 映射到 $y$ 的函数 (Goal: Learn a function to map x -> y) [2:40]
        -   例子: 图像分类 (Image Classification)、回归 (Regression)、目标检测 (Object Detection)、语义分割 (Semantic Segmentation)、图像标注 (Image Captioning) 等 [3:30]
        -   核心特点: 需要人工标注数据 (Requires human annotation) [2:54]
    -   **无监督学习 (Unsupervised Learning)**
        -   数据形式: 仅有数据 $x$，没有标签 (Data: x, Just data, no labels!) [4:30]
        -   目标: 学习数据中潜在的隐藏结构 (Goal: Learn some underlying hidden structure of the data) [5:00]
        -   例子: 聚类 (Clustering) (如 K-Means)、降维 (Dimensionality Reduction) (如主成分分析 PCA)、特征学习 (Feature Learning) (如自编码器 Autoencoders)、密度估计 (Density Estimation) [6:00]
        -   核心特点: 不需要人工标注 (Doesn't require human annotation)，被视为机器学习的“圣杯” (Holy Grail) [5:17]
-   **判别模型 vs. 生成模型 (Discriminative vs. Generative Models)** [9:09]
    -   **判别模型 (Discriminative Model)**
        -   学习条件概率分布 $p(y|x)$ (Learn a probability distribution $p(y|x)$) [9:57]
        -   特点: 对于每个输入，可能的标签之间竞争概率质量 (Possible labels for each input "compete" for probability mass) [14:45]
        -   局限性: 无法处理不合理的输入，因为它必须为所有图像给出标签分布 (No way for the model to handle unreasonable inputs; it must give label distributions for all images) [14:57]
        -   这可能是对抗性攻击 (Adversarial Attacks) 可能的原因之一 [15:51]
    -   **生成模型 (Generative Model)**
        -   学习数据 $x$ 的概率分布 $p(x)$ (Learn a probability distribution $p(x)$) [10:26]
        -   特点: 所有可能的图像之间竞争概率质量 (All possible images compete with each other for probability mass) [16:40]
        -   优势: 可以“拒绝”不合理的输入，通过为其分配非常小的值 (Can "reject" unreasonable inputs by assigning them small values) [20:00]
        -   需要对图像有深刻的理解 (Requires deep image understanding) [17:09]
    -   **条件生成模型 (Conditional Generative Model)**
        -   学习给定标签 $y$ 下数据 $x$ 的概率分布 $p(x|y)$ (Learn $p(x|y)$) [10:32]
        -   特点: 每个可能的标签都会引起所有图像之间的竞争 (Each possible label induces a competition among all images) [20:41]
        -   可以分配标签，同时拒绝异常值 (Assign labels, while rejecting outliers!) [21:45]
        -   可以生成以输入标签为条件的新数据 (Generate new data conditioned on input labels) [25:46]
    -   **概率回顾 (Probability Recap)**
        -   密度函数 $p(x)$ (Density Function $p(x)$): 为每个可能的 $x$ 分配一个正数，数值越高表示 $x$ 越可能 (assigns a positive number to each possible $x$; higher numbers mean $x$ is more likely) [11:16]
        -   密度函数是归一化的: $\int_{X} p(x) dx = 1$ (Density functions are normalized) [12:00]
        -   不同的 $x$ 值竞争密度 (Different values of x compete for density) [12:44]
    -   **贝叶斯定理 (Bayes' Rule)**
        -   $P(x|y) = \frac{P(y|x)}{P(y)} P(x)$ [22:33]
        -   这表明可以从判别模型 (Discriminative Model, $P(y|x)$)、标签的先验分布 (Prior over labels, $P(y)$) 和无条件生成模型 (Unconditional Generative Model, $P(x)$) 构建条件生成模型 (Conditional Generative Model, $P(x|y)$) [22:50]
-   **生成模型分类 (Taxonomy of Generative Models)** [26:19]
    -   **显式密度模型 (Explicit density)**: 模型可以计算 $p(x)$ (Model can compute $p(x)$) [26:43]
        -   **可处理密度 (Tractable density)**: 可以直接计算 $p(x)$ (Can compute $p(x)$) [27:30]
            -   自回归模型 (Autoregressive models) [28:45, 30:05]
            -   NADE (Neural Autoregressive Distribution Estimator) / MADE (Masked Autoencoder for Density Estimation)
            -   NICE (Non-linear Independent Components Estimation) / RealNVP (Real NVP)
            -   Glow (Generative Flow with Invertible 1x1 Convolutions)
            -   FFjord (Free-form Jacobian of Flows with Ordinary Differential Equations)
        -   **近似密度 (Approximate density)**: 可以计算 $p(x)$ 的近似值 (Can compute approximation to $p(x)$) [27:43]
            -   变分方法 (Variational): 变分自编码器 (Variational Autoencoder) [28:03]
            -   马尔可夫链 (Markov Chain): GSN (Generative Stochastic Networks), Boltzmann Machine
    -   **隐式密度模型 (Implicit density)**: 模型不显式计算 $p(x)$，但可以从中采样 (Model does not explicitly compute $p(x)$, but can sample from $p(x)$) [27:14]
        -   马尔可夫链 (Markov Chain): GSN
        -   直接 (Direct): 生成对抗网络 (Generative Adversarial Networks (GANs)) [28:11]
-   **显式密度估计：自回归模型 (Explicit Density: Autoregressive Models)** [30:12]
    -   目标 (Goal): 写出 $p(x) = f(x, W)$ 的显式函数 [30:12]
    -   假设 $x$ 包含多个子部分 (Assume x consists of multiple subparts): $x = (x_1, x_2, x_3, ..., x_T)$ [32:37]
    -   使用链式法则分解概率 (Break down probability using the chain rule): $p(x) = p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)... = \prod_{t=1}^{T} p(x_t | x_1, ..., x_{t-1})$ [32:58]
        -   这表示下一个子部分给定所有前一个子部分的概率 (Probability of the next subpart given all the previous subparts) [33:40]
        -   这种模式与循环神经网络 (RNN) 中的语言建模 (Language modeling) 类似 [34:07]
    -   **PixelRNN** [35:30]
        -   从图像左上角开始，一次生成一个图像像素 (Generate image pixels one at a time, starting at the upper left corner) [35:42]
        -   为每个像素计算一个隐藏状态 (Compute a hidden state for each pixel)
        -   隐藏状态取决于左侧和上方像素的隐藏状态和 RGB 值 (depends on hidden states and RGB values from the left and from above (LSTM recurrence)) [35:53]
            -   $h_{x,y} = f(h_{x-1,y}, h_{x,y-1}, W)$ [35:53]
        -   在每个像素处，预测红色，然后蓝色，然后绿色 (At each pixel, predict red, then blue, then green): 对 [0, 1, ..., 255] 进行 Softmax [36:07]
        -   每个像素隐式依赖于其上方和左侧的所有像素 (Each pixel depends implicitly on all pixels above and to the left) [37:32]
        -   **问题 (Problem)**: 在训练和测试期间都非常慢 (Very slow during both training and testing) [38:20]
            -   N x N 图像需要 2N-1 个顺序步骤 (N x N image requires 2N-1 sequential steps) [38:43]
    -   **PixelCNN** [38:58]
        -   仍然从角落开始生成图像像素 (Still generate image pixels starting from corner) [39:07]
        -   对先前像素的依赖现在使用上下文区域上的 CNN 进行建模 (Dependency on previous pixels now modeled using a CNN over context region) [39:19]
        -   训练: 最大化训练图像的似然 (Training: maximize likelihood of training images) [39:41]
        -   训练速度比 PixelRNN 快 (Training is faster than PixelRNN)
            -   可以并行化卷积，因为训练图像中的上下文区域值是已知的 (can parallelize convolutions since context region values known from training images) [39:50]
        -   生成仍然需要按顺序进行 (Generation must still proceed sequentially) => 仍然很慢 (still slow) [39:56]
    -   **PixelRNN: 生成样本 (Generated Samples)** [40:00]
        -   32x32 CIFAR-10 和 32x32 ImageNet 的生成图像 (examples shown) [40:00]
        -   图片看起来模糊，包含一些高层结构，但细节仍有“垃圾” (Images look blurry, contain some high-level structures, but details are "garbage") [40:47]
    -   **自回归模型：PixelRNN 和 PixelCNN 的优缺点 (Autoregressive Models: PixelRNN and PixelCNN Pros & Cons)** [40:57]
        -   **优点 (Pros)**:
            -   可以显式计算 $p(x)$ 的似然 (Can explicitly compute likelihood $p(x)$) [41:06]
            -   训练数据的显式似然提供了良好的评估指标 (Explicit likelihood of training data gives good evaluation metric) [41:14]
            -   样本质量良好 (Good samples) [41:50]
        -   **缺点 (Con)**:
            -   顺序生成 (Sequential generation) => 速度慢 (slow) [43:40]
        -   **改进 PixelCNN 性能 (Improving PixelCNN performance)**:
            -   门控卷积层 (Gated convolutional layers)
            -   短连接 (Short-cut connections)
            -   离散逻辑损失 (Discretized logistic loss)
            -   多尺度 (Multi-scale)
            -   训练技巧 (Training tricks)
            -   等等 (Etc...)
        -   **相关论文 (See)**:
            -   Van den Oord et al., "Pixel Recurrent Neural Networks", ICML 2016 [40:57]
            -   Salimans et al. 2017 (PixelCNN++) [40:57]

-   **变分自编码器 (Variational Autoencoders - VAE)** [44:23]
    -   PixelRNN / PixelCNN 显式地使用神经网络参数化密度函数，因此可以训练以最大化训练数据的似然 (PixelRNN / PixelCNN explicitly parameterizes density function with a neural network, so we can train to maximize likelihood of training data) [44:40]
    -   变分自编码器 (VAE) 定义了一个**难处理的密度**，我们无法显式计算或优化它 (Variational Autoencoders (VAE) define an **intractable density** that we cannot explicitly compute or optimize) [45:10]
    -   但我们可以直接优化密度的**下界** (But we will be able to directly optimize a **lower bound** on the density) [45:24]

-   **常规（非变分）自编码器 (Regular, non-variational Autoencoders)** [46:20]
    -   一种无监督学习方法，用于从原始数据 $x$ 中学习特征向量 $z$，无需任何标签 (Unsupervised method for learning feature vectors from raw data x, without any labels) [46:21]
    -   特征 (Features) 应该提取有用的信息 (should extract useful information) (例如对象身份、属性、场景类型等)，以便用于下游任务 (that we can use for downstream tasks) [46:57]
    -   **编码器 (Encoder)**: 从输入数据 $x$ 到特征 $z$ 的映射 (map $x$ to $z$) [47:00]
        -   最初是线性 + 非线性（sigmoid）(Originally: Linear + nonlinearity (sigmoid)) [47:58]
        -   后来是深度全连接 (Later: Deep, fully-connected) [48:07]
        -   再后来是 ReLU CNN (Later: ReLU CNN) [48:16]
    -   **问题 (Problem)**: 我们如何从原始数据中学习这种特征转换？我们无法观察到特征！ (How can we learn this feature transform from raw data? But we can't observe features!) [47:33]
    -   **思路 (Idea)**: 使用特征通过解码器 (decoder) 重构输入数据 (Use the features to reconstruct the input data with a decoder) [48:26]
        -   “自编码 (Autoencoding)” = 自我编码 (encoding itself) [48:32]
        -   **解码器 (Decoder)**: 从特征 $z$ 到重构输入数据 $\hat{x}$ 的映射 (map $z$ to $\hat{x}$) [48:45]
            -   架构类似编码器，但通常是“翻转”的版本（例如，CNN 使用上采样或转置卷积层 (upconv layers)）[48:58]
    -   **损失函数 (Loss Function)**: 输入数据和重构数据之间的 L2 距离 (L2 distance between input and reconstructed data) [49:09]
        -   不使用任何标签！仅原始数据！ (Does not use any labels! Just raw data!) [49:10]
        -   $\text{Loss Function} = ||\hat{x} - x||^2_2$ [49:09]
    -   **训练后 (After training)**: 扔掉解码器 (throw away decoder) 并使用编码器进行下游任务 (use encoder for a downstream task) [51:21]
        -   编码器可以用于初始化监督模型 (Encoder can be used to initialize a supervised model) [51:31]
        -   在少量数据上训练最终任务 (Train for final task (sometimes with small data)) [51:41]
    -   **局限性 (Limitations)**:
        -   自编码器学习数据的**潜在特征** (Autoencoders learn **latent features** for data without any labels!) [53:27]
        -   可以用特征初始化监督模型 (Can use features to initialize a supervised model) [53:28]
        -   **非概率性 (Not probabilistic)**: 无法从学习到的模型中采样新数据 (No way to sample new data from learned model) [53:38]

-   **变分自编码器 (Variational Autoencoders - VAE)** [54:52]
    -   自编码器的概率化版本 (Probabilistic spin on autoencoders) [54:53]
    -   1. 从原始数据中学习潜在特征 $z$ (Learn latent features z from raw data) [55:08]
    -   2. 从模型中采样以生成新数据 (Sample from the model to generate new data) [55:16]
    -   **直觉 (Intuition)**: $x$ 是一张图像，$z$ 是用于生成 $x$ 的潜在因子 (x is an image, z is latent factors used to generate x): 属性、方向等 (attributes, orientation, etc.) [55:38]
    -   假设训练数据 $\{x^{(i)}\}_{i=1}^N$ 是由未观察到的（潜在）表示 $z$ 生成的 (Assume training data is generated from unobserved (latent) representation z) [55:24]
    -   假设一个简单的先验 $p(z)$，例如高斯分布 (Assume simple prior p(z), e.g. Gaussian) [56:39]
    -   用神经网络表示条件概率 $p(x|z)$ (Represent $p(x|z)$ with a neural network) [56:58]
        -   类似于自编码器中的解码器 (Similar to decoder from autoencoder) [57:00]
        -   解码器 (Decoder) 必须是**概率性**的 (must be probabilistic) [57:21]
        -   解码器输入 $z$，输出高斯分布的均值 $\mu_{x|z}$ 和（对角线）协方差 $\Sigma_{x|z}$ (Decoder inputs z, outputs mean $\mu_{x|z}$ and (diagonal) covariance $\Sigma_{x|z}$) [57:39]
        -   从高斯分布中采样 $x$，其均值和协方差由解码器网络输出 (Sample $x$ from Gaussian with mean $\mu_{x|z}$ and (diagonal) covariance $\Sigma_{x|z}$) [57:46]
    -   **如何训练这个模型？(How to train this model?)** [1:00:04]
        -   基本思想: 最大化数据的似然 (Basic idea: maximize likelihood of data) [1:00:11]
        -   我们没有观察到 $z$，所以需要进行边缘化 (We don't observe $z$, so need to marginalize):
            $$p_\theta(x) = \int p_\theta(x,z)dz = \int p_\theta(x|z)p_\theta(z)dz$$
            
            -   $p_\theta(x|z)$ 由解码器网络计算 (computed by decoder network) [1:01:29]
            -   $p_\theta(z)$ 是我们假设的先验高斯分布 (assumed Gaussian prior) [1:01:33]
        -   **问题 (Problem)**: 这个积分无法计算 (Impossible to integrate over all z!) [1:01:37]
    -   **另一个思路: 尝试贝叶斯法则 (Another idea: Try Bayes' Rule)** [1:01:50]
        -   $\log p_\theta(x) = \log \frac{p_\theta(x|z)p_\theta(z)}{p_\theta(z|x)}$ [1:02:15]
        -   乘以并除以 $q_\phi(z|x)$ (Multiply top and bottom by $q_\phi(z|x)$) (这是一个新的网络，编码器网络，其参数为 $\phi$) [1:02:21]
        -   编码器网络 $q_\phi(z|x)$ 输入数据 $x$，给出潜在编码 $z$ 的分布 (Encoder network inputs data x, gives distribution over latent codes z) [1:04:26]
            -   $q_\phi(z|x) = N(\mu_{z|x}, \Sigma_{z|x})$ [1:04:31]
            -   编码器输入 $x$，输出高斯分布的均值 $\mu_{z|x}$ 和（对角线）协方差 $\Sigma_{z|x}$ (Encoder inputs x, outputs mean $\mu_{z|x}$ and (diagonal) covariance $\Sigma_{z|x}$) [1:04:31]
        -   如果我们可以确保 $q_\phi(z|x) \approx p_\theta(z|x)$ (If we can ensure that $q_\phi(z|x) \approx p_\theta(z|x)$)，那么我们可以近似 (then we can approximate):
            $$p_\theta(x) \approx \frac{p_\theta(x|z)p_\theta(z)}{q_\phi(z|x)}$$

        -   我们可以用期望值来重写这个对数似然项 (We can rewrite this log-likelihood term as an expectation) [1:06:01]
            -   $\log p_\theta(x) = E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z)) + D_{KL}(q_\phi(z|x) || p_\theta(z|x))$ [1:07:15]
            -   $D_{KL}(P || Q)$ 是 KL 散度 (KL divergence) [1:07:30]
        -   我们知道 $D_{KL}(Q || P) \ge 0$ (We know $D_{KL}(Q || P) \ge 0$) [1:08:21]
        -   因此，我们得到一个**数据似然的下界** (Therefore, we get a lower bound on the data likelihood):
            $$\log p_\theta(x) \ge E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$ [1:08:44]
            -   这个下界被称为**变分下界** (variational lower bound) [1:09:27]
            -   第一项是**数据重构**项 (Data reconstruction) [1:09:02]
            -   第二项是**KL 散度** (KL divergence) [1:09:02]
    -   **联合训练编码器 q 和解码器 p 以最大化变分下界 (Jointly train encoder q and decoder p to maximize the variational lower bound)** [1:09:12]
        -   在每次训练迭代中，我们对 $z$ 从 $q_\phi(z|x)$ 中进行采样，并使用这些采样值计算损失，然后通过反向传播更新 $\theta$ 和 $\phi$。
        -   这使编码器能够学习如何将数据映射到潜在空间，同时解码器学习如何从潜在空间生成数据。
        -   **关键 (Key)**: 这种方法使得在难处理的概率模型上进行优化成为可能，同时能够生成高质量的样本。
        -   **优缺点 (Pros & Cons)**:
            -   VAE 的样本通常比 GAN 模糊 (VAEs samples are usually blurrier than GANs)
            -   然而，它们提供了似然估计，这是一个重要的评估指标 (But they provide likelihood estimation, which is an important evaluation metric)

本次视频通过数学公式和概念推导展示了算法的核心逻辑，但未包含具体的 Python 代码示例。

### 二、关键术语定义 (Key Term Definitions)
-   **生成模型 (Generative Models)**: 一类机器学习模型，旨在学习训练数据的底层分布，从而能够生成与训练数据相似的新数据样本。
-   **监督学习 (Supervised Learning)**: 一种机器学习范式，模型通过带有输入-输出对（数据和对应标签）的标注数据进行训练，以学习从输入到输出的映射关系。
-   **无监督学习 (Unsupervised Learning)**: 一种机器学习范式，模型在没有明确标注数据的情况下，通过识别数据中的模式和结构进行学习。
-   **图像分类 (Image Classification)**: 监督学习的一个应用，目标是将图像归类到预定义的类别中。
-   **回归 (Regression)**: 监督学习的一个应用，目标是预测一个连续的输出值。
-   **目标检测 (Object Detection)**: 监督学习的一个应用，目标是在图像中识别并定位物体。
-   **语义分割 (Semantic Segmentation)**: 监督学习的一个应用，目标是对图像中的每个像素进行分类。
-   **图像标注 (Image Captioning)**: 监督学习的一个应用，目标是为图像生成自然语言描述。
-   **聚类 (Clustering)**: 无监督学习的一个应用，目标是将数据点分组，使得同组内的数据点相似度高，不同组间相似度低。
-   **降维 (Dimensionality Reduction)**: 无监督学习的一个应用，目标是将高维数据映射到低维空间，同时保留其大部分重要信息。
-   **特征学习 (Feature Learning)**: 无监督学习的一个应用，目标是自动发现数据中的有效表示或特征。
-   **密度估计 (Density Estimation)**: 无监督学习的一个应用，目标是估计数据点的概率密度函数。
-   **判别模型 (Discriminative Model)**: 机器学习模型的一种，直接学习条件概率 $P(y|x)$，用于区分不同类别，而不显式建模数据的生成过程。
-   **条件生成模型 (Conditional Generative Model)**: 一种生成模型，学习给定特定条件（如标签）下数据的概率分布 $P(x|y)$。
-   **密度函数 (Density Function)**: 概率论中的一个函数，它描述了随机变量在给定点附近的相对可能性。对于连续变量，它表示概率密度，其积分在整个空间上等于1。
-   **贝叶斯定理 (Bayes' Rule)**: 一条在概率论中用于计算条件概率的定理，其公式为 $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$。在机器学习中常用于连接不同类型的模型。
-   **显式密度 (Explicit Density)**: 指那些能够明确计算出其概率密度函数 $p(x)$ 的生成模型。
-   **隐式密度 (Implicit Density)**: 指那些不能直接计算其概率密度函数 $p(x)$，但可以从中进行采样的生成模型。
-   **可处理密度 (Tractable Density)**: 显式密度模型的一个子类别，其概率密度函数 $p(x)$ 可以高效精确地计算。
-   **近似密度 (Approximate Density)**: 显式密度模型的一个子类别，其概率密度函数 $p(x)$ 只能通过近似方法来计算。
-   **变分方法 (Variational Methods)**: 用于近似难以处理的概率分布或推断的方法，通常通过优化一个下界来实现。
-   **变分自编码器 (Variational Autoencoder - VAE)**: 一种基于变分方法的生成模型，它学习数据的潜在表示并能从中生成新的数据。
-   **马尔可夫链 (Markov Chain)**: 一种随机过程，其中每个状态的概率只取决于前一个状态。在生成模型中用于建模序列数据或采样。
-   **Boltzmann Machine (玻尔兹曼机)**: 一种基于马尔可夫链的能量模型，用于学习复杂数据的分布。
-   **生成对抗网络 (Generative Adversarial Networks - GANs)**: 一种隐式密度模型，由一个生成器和一个判别器组成，通过对抗训练学习生成逼真的数据。
-   **自回归模型 (Autoregressive Models)**: 一种可处理密度的生成模型，它通过将数据的联合概率分解为一系列条件概率的乘积来建模数据，其中每个数据点都以前面的数据点为条件。
-   **NADE (Neural Autoregressive Distribution Estimator)**: 一种使用神经网络实现自回归模型的显式密度估计器。
-   **MADE (Masked Autoencoder for Density Estimation)**: 另一种使用掩码自编码器实现自回归模型的显式密度估计器。
-   **NICE (Non-linear Independent Components Estimation)**: 一种使用可逆变换的显式密度模型，可以精确计算密度和采样。
-   **RealNVP (Real NVP)**: NICE 的改进版本，同样使用可逆变换来实现精确的密度估计和采样。
-   **Glow (Generative Flow with Invertible 1x1 Convolutions)**: 基于流模型 (Flow-based model) 的生成模型，可以实现可逆的生成和密度估计。
-   **FFjord (Free-form Jacobian of Flows with Ordinary Differential Equations)**: 一种基于ODE的流模型，能够学习复杂的概率分布。
-   **最大似然估计 (Maximum Likelihood Estimation)**: 一种用于估计模型参数的方法，通过找到使观测数据的概率（或似然）最大化的参数值。
-   **Log Trick (对数技巧)**: 在最大似然估计中，为了将乘积转化为和，从而简化计算和避免数值下溢，常对似然函数取对数。
-   **链式法则 (Chain Rule)**: 概率论中用于分解联合概率分布的规则，即将一个联合分布表示为一系列条件分布的乘积。
-   **循环神经网络 (Recurrent Neural Network - RNN)**: 一种适用于序列数据的神经网络，其特点是信息可以在网络中循环，使得当前输出依赖于过去的输入。
-   **PixelRNN**: 一种自回归的显式密度模型，通过顺序预测图像像素（包括 RGB 通道），以其左侧和上方像素为条件来生成图像。
-   **PixelCNN**: PixelRNN 的变体，利用卷积神经网络（CNN）来建模像素间的依赖关系，尤其是在上下文区域，以实现更快的训练速度。
-   **自编码器 (Autoencoder)**: 一种无监督神经网络，旨在学习数据的有效编码（潜在特征），通过尝试将输入重构为输出。
-   **编码器 (Encoder)**: 自编码器的一部分，负责将输入数据映射到潜在特征空间。
-   **解码器 (Decoder)**: 自编码器的一部分，负责将潜在特征映射回原始数据空间，尝试重构输入。
-   **变分下界 (Variational Lower Bound - ELBO)**: 变分自编码器中优化的目标函数，它是数据似然的一个下界，通过最大化此下界来间接最大化数据的似然。
-   **KL 散度 (KL Divergence)**: 衡量两个概率分布之间差异的非负度量。在 VAE 中，它通常用于衡量近似后验分布与先验分布之间的距离。
-   **变分推断 (Variational Inference)**: 一种用于近似复杂概率分布或难以处理的推断问题的方法，通过引入一个更简单的变分分布来近似目标分布，并优化其参数。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **显式密度估计 (Explicit Density Estimation)**
    -   **目标 (Goal)**: 写出 $p(x) = f(x, W)$ 的显式函数。
    -   **训练方法 (Training Method)**: 给定数据集 $x^{(1)}, x^{(2)}, ..., x^{(N)}$，通过解决以下优化问题来训练模型：
        $$W^* = \arg \max_W \prod_i p(x^{(i)})$$
        -   这旨在最大化训练数据的概率（即最大似然估计，Maximum Likelihood Estimation）。
        -   为了便于优化，通常使用对数技巧 (Log trick) 将乘积转换为和：
        $$W^* = \arg \max_W \sum_i \log p(x^{(i)})$$
        -   将 $p(x^{(i)})$ 替换为我们的参数化函数 $f(x^{(i)}, W)$：
        $$W^* = \arg \max_W \sum_i \log f(x^{(i)}, W)$$
        -   这个表达式将成为我们的损失函数 (loss function)，可以使用梯度下降 (gradient descent) 进行训练。

-   **自回归模型 (Autoregressive Models)**
    -   **核心思想**: 假设每个数据点 $x$ 由多个子部分组成：$x = (x_1, x_2, x_3, ..., x_T)$。
    -   **概率分解 (Probability Breakdown)**: 利用链式法则 (Chain rule) 将联合概率分解为条件概率的乘积：
        $$p(x) = p(x_1, x_2, x_3, ..., x_T) = p(x_1)p(x_2 | x_1)p(x_3 | x_1, x_2)...$$
        $$= \prod_{t=1}^{T} p(x_t | x_1, ..., x_{t-1})$$
        -   这表示下一个子部分给定所有前一个子部分的概率。
    -   **PixelRNN 像素生成过程**:
        -   从左上角开始，依次生成每个像素的 RGB 值。
        -   当前像素的隐藏状态 $h_{x,y}$ 依赖于左侧和上方像素的隐藏状态 $h_{x-1,y}$ 和 $h_{x,y-1}$，以及它们对应的 RGB 值。
        -   $h_{x,y} = f(h_{x-1,y}, h_{x,y-1}, W)$
        -   每个像素的 RGB 值通过 Softmax 输出其在 范围内的离散概率分布。
        -   这种依赖关系通过 RNN 的序列特性自然地实现了对所有先前像素的隐式条件依赖。
    -   **PixelCNN 像素生成过程**:
        -   与 PixelRNN 类似，但使用掩码卷积 (masked convolutions) 来强制依赖关系，只考虑左侧和上方的像素信息。
        -   这允许在训练时并行计算某些卷积操作，从而加快训练速度，但生成过程仍然是顺序的。

-   **变分自编码器 (Variational Autoencoders - VAE)**
    -   **目标**: 学习数据的潜在特征 $z$ 并能够从模型中采样以生成新数据。
    -   **模型结构**:
        -   **解码器网络 (Decoder Network)**：输入潜在代码 $z$，输出数据 $x$ 的概率分布。
            -   $p_\theta(x|z) = N(\mu_{x|z}, \Sigma_{x|z})$
            -   解码器网络输出高斯分布的均值 $\mu_{x|z}$ 和（对角线）协方差 $\Sigma_{x|z}$。
            -   然后从这个高斯分布中采样 $x$。
        -   **编码器网络 (Encoder Network)**（又称**变分推断网络**）：输入数据 $x$，输出潜在代码 $z$ 的分布。
            -   $q_\phi(z|x) = N(\mu_{z|x}, \Sigma_{z|x})$
            -   编码器网络输出高斯分布的均值 $\mu_{z|x}$ 和（对角线）协方差 $\Sigma_{z|x}$。
            -   然后从这个高斯分布中采样 $z$。
    -   **训练目标**: 最大化数据的似然 $p_\theta(x)$。由于直接计算此积分难处理，我们转而最大化其**变分下界 (Variational Lower Bound - ELBO)**。
        -   **数据似然的分解**:
            $$ \log p_\theta(x) = E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) + D_{KL}(q_\phi(z|x) || p_\theta(z|x)) $$
            -   $E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]$：数据重构项 (Data reconstruction term)，衡量解码器从潜在代码重构输入数据的能力。
            -   $D_{KL}(q_\phi(z|x) || p_\theta(z))$：KL 散度项，衡量编码器输出的近似后验分布与潜在变量先验分布之间的距离。
            -   $D_{KL}(q_\phi(z|x) || p_\theta(z|x))$：KL 散度项，衡量编码器输出的近似后验分布与真实后验分布之间的距离。此项通常无法直接计算，但因其非负性，可以被省略以获得下界。
        -   **变分下界 (ELBO)**:
            $$ \log p_\theta(x) \ge E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z)) $$
            -   我们通过联合训练编码器和解码器来最大化这个下界。

本次视频未包含具体的 Python 代码示例。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   “关于监督学习和无监督学习的区别，大家清楚了吗？有什么问题吗？” (Was this clear to everyone? Was any kind of questions on this supervised versus unsupervised uh the distinction?) [8:56]
-   “我们需要使用监督学习来训练条件生成模型吗？” (Do we need to use supervised learning to learn the conditional generative model?) [11:39]
-   “我们如何判断一个生成模型的好坏？” (How can we tell how good is a generative model?) [19:16]
-   “这看起来像是什么东西？” (Can anyone guess what this reminds you of?) [34:01] (指链式法则分解概率的模式)
-   “对于这种一次生成一个像素的模型，我们必须生成第一个像素吗？” (For these kinds of models where we're generating one pixel at a time, do we have to generate the first pixel?) [42:17]
-   “这些模型能推广到不同的图像分辨率吗？” (Can these models generalize to different image resolutions?) [43:51]
-   “如果有一个编码器但有不同的解码器，那会怎么样？” (If you had one encoder but different decoders, what would that be?) [52:43]
-   “如果我们有1个编码器但有不同的解码器，那会怎么样？” (What if you have one encoder but multiple decoders?) [52:43]

---
### [📚] 视频学习脚手架: Generative Models, Part 2

### 一、核心内容大纲 (Core Content Outline)

-   **引言 (Introduction)**
    -   继续生成模型 (Generative Models) 的讨论。

-   **监督学习 (Supervised Learning) 与无监督学习 (Unsupervised Learning) 回顾**
    -   **监督学习 (Supervised Learning)**
        -   数据: $(x, y)$，其中 $x$ 为数据， $y$ 为标签 (label)。
        -   目标 (Goal): 学习一个函数 (function) 将 $x$ 映射到 $y$。
        -   示例 (Examples): 分类 (classification)、回归 (regression)、目标检测 (object detection)、语义分割 (semantic segmentation)、图像标注 (image captioning)。
        -   特点: 需要带有标签的数据 (labeled data)。
    -   **无监督学习 (Unsupervised Learning)**
        -   数据: 只有 $x$，没有标签。
        -   目标 (Goal): 学习数据中潜在的隐藏结构 (underlying hidden structure)。
        -   示例 (Examples): 聚类 (clustering)、降维 (dimensionality reduction)、特征学习 (feature learning)、密度估计 (density estimation)。
        -   挑战: 被认为是机器学习领域的“圣杯”问题，即如何从无标签的海量数据中学习有用的表示。

-   **判别模型 (Discriminative Model) 与生成模型 (Generative Model) 回顾**
    -   **判别模型 (Discriminative Model)**
        -   学习概率分布 (probability distribution) $p(y|x)$。
        -   目标: 建模给定输入 $x$ 时，输出 $y$ 的概率。
        -   特点: 概率分布必须归一化 (normalized)，导致不同标签之间存在竞争 (competition)。
        -   缺点: 无法处理不合理的输入，模型必须为所有图像给出标签分布。
    -   **生成模型 (Generative Model)**
        -   学习概率分布 $p(x)$。
        -   目标: 为每个可能的输入 $x$ 分配一个正数，数值越高表示 $x$ 越可能。
        -   特点: 所有可能的图像都在争夺概率质量 (probability mass)。
        -   要求: 需要对数据类型有非常深入的理解。
    -   **条件生成模型 (Conditional Generative Model)**
        -   学习概率 $p(x|y)$。
        -   与判别模型和无条件生成模型的关系 (Recall Bayes' Rule): $P(x|y) = \frac{P(y|x) \cdot P(x)}{P(y)}$。

-   **生成模型分类 (Taxonomy of Generative Models)**
    -   模型能计算 $p(x)$ (显式密度 - Explicit Density)
        -   **可处理密度 (Tractable Density)**: 可以计算 $p(x)$。
            -   **自回归模型 (Autoregressive Models)**: 如 PixelRNN, NADE/MADE, NICE/RealNVP, Glow, Ffjord。
        -   **近似密度 (Approximate Density)**: 可以计算 $p(x)$ 的近似值。
            -   **变分模型 (Variational Models)**: 如变分自编码器 (Variational Autoencoder - VAE)。
            -   **马尔可夫链 (Markov Chain)**: 如玻尔兹曼机 (Boltzmann Machine)。
    -   模型不显式计算 $p(x)$，但可以从中采样 (Implicit Density)。
        -   **马尔可夫链 (Markov Chain)**: 如 GSN。
        -   **直接 (Direct)**: 如生成对抗网络 (Generative Adversarial Networks - GANs)。

-   **自回归模型 (Autoregressive Models)**
    -   **显式密度函数 (Explicit Density Function)**: $p(x) = \prod_{t=1}^{T} p(x_t | x_1, \dots, x_{t-1})$
    -   训练 (Train): 通过最大化训练数据 (training data) 的对数似然 (log-likelihood) 进行。
    -   特点:
        -   可以直接最大化数据概率 (p(data))。
        -   生成高质量 (High-quality)、清晰 (sharp) 的图像。
        -   生成图像速度慢 (Slow to generate images)。
        -   没有显式潜在代码 (No explicit latent codes)。
    -   示例: PixelRNN (像素循环神经网络) 和 PixelCNN (像素卷积神经网络)。

-   **变分自编码器 (Variational Autoencoders - VAEs)**
    -   **核心思想**: 引入潜在变量 (latent variable) $z$，并最大化数据似然 (data likelihood) 的变分下限 (variational lower bound)。
    -   **训练目标 (Training Objective)**: 联合训练编码器 (encoder) $q_{\phi}(z|x)$ 和解码器 (decoder) $p_{\theta}(x|z)$。
        -   $\log p_{\theta}(x) \ge E_{Z \sim q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x), p(z))$
        -   $D_{KL}(q_{\phi}(z|x), p(z))$ 是编码器输出分布 $q_{\phi}(z|x)$ 与先验分布 $p(z)$ 之间的KL散度。
        -   KL散度鼓励编码器学习到的潜在分布接近预设的先验分布（通常是单位高斯分布）。
        -   通过重参数化技巧 (reparameterization trick) 从 $q_{\phi}(z|x)$ 采样 $z$。
    -   **优点 (Pros)**:
        -   生成模型的一种有原则的方法 (Principled approach)。
        -   允许推断 (inference) $q(z|x)$，可作为其他任务的有用特征表示 (useful feature representation)。
        -   生成图像速度快 (Very fast to generate images)。
        -   学习到丰富且有意义的潜在代码 (Learn rich latent codes)。
        -   通过先验分布 $p(z)$（如对角高斯分布 - diagonal Gaussian）强制潜在维度独立，实现“解缠变异因子 (Disentangling factors of variation)”。
    -   **缺点 (Cons)**:
        -   最大化的是似然的下限 (Maximizes lower bound of likelihood)，评价指标不如直接最大化似然的模型（如 PixelRNN/PixelCNN）。
        -   生成的样本通常比较模糊 (blurry) 且质量较低 (lower quality)。
    -   **图像生成与编辑 (Image Generation and Editing)**:
        -   通过修改潜在代码 (latent code) $z$ 的特定维度来编辑图像的语义属性，如“微笑程度 (degree of smile)”（对应 z1）和“头部姿态 (head pose)”（对应 z2）。
        -   模型自动学习这些维度与语义属性的对应关系，我们无法预先指定。
        -   示例: 通过变分自编码器对人脸图像进行编辑，改变姿态 (Pose / Azimuth) 或光照方向 (Light direction)。
    -   **活跃研究领域 (Active areas of research)**:
        -   更灵活的近似 (More flexible approximations)，例如更丰富的近似后验分布 (richer approximate posterior) 代替对角高斯分布 (diagonal Gaussian)，如高斯混合模型 (Gaussian Mixture Models - GMMs)。
        -   在潜在变量中引入结构 (Incorporating structure in latent variables)，例如分类分布 (Categorical Distributions)。

-   **结合 VAE 与自回归模型 (Combining VAE + Autoregressive)**
    -   **目标 (Goal)**: 结合自回归模型的高质量生成能力和 VAE 的快速采样与丰富潜在代码的优点。
    -   **VQ-VAE2 (Vector-Quantized Variational Autoencoder 2)**:
        -   **训练方式**: 训练一个 VAE 类型的模型来生成多尺度 (multiscale) 的潜在代码网格 (grids of latent codes)。
            -   编码器将原始图像编码成不同级别的潜在代码网格（如 Top Level 和 Bottom Level）。
            -   解码器从这些潜在代码网格重构图像。
        -   **采样方式**: 使用多尺度 PixelCNN 在量化 (quantized) 后的潜在代码空间中进行采样。
            -   PixelCNN 在潜在代码空间而非原始像素空间操作，从而加速采样过程。
        -   **结果**: 能够生成非常高质量、高保真度的图像（如 256x256 的 ImageNet 图像，1024x1024 的逼真人脸图像）。这些图像是模型生成的“假人”，而不是真实存在的。

-   **生成对抗网络 (Generative Adversarial Networks - GANs)**
    -   **思想 (Idea)**: 放弃显式建模数据概率密度 $p(x)$，转而学习从 $p(x)$ 中采样 (draw samples) 的能力。
    -   **基本设置 (Setup)**:
        -   假设有数据 $x_i$ 从真实数据分布 $p_{data}(x)$ 中采样。
        -   目标是学习一个模型，能够从 $p_{data}(x)$ 中采样新的 $x$。
        -   引入一个潜在变量 $z$（通常是简单的先验分布，如高斯或均匀分布）。
        -   通过一个**生成器网络 (Generator Network)** $G(z)$ 将 $z$ 映射到数据空间 $x$。
        -   $G(z)$ 生成的样本分布称为生成器分布 $p_G$。目标是使 $p_G = p_{data}$。
    -   **训练目标 (Training Objective) - Minimax Game**:
        -   引入一个**判别器网络 (Discriminator Network)** $D(x)$。
        -   $D(x)$ 的任务是区分输入图像是来自真实数据（标签为1）还是来自生成器（标签为0）。
        -   **判别器的目标 (Discriminator's Goal)**: 最大化正确分类的概率。
        -   **生成器的目标 (Generator's Goal)**: 最小化判别器正确分类的概率，即“欺骗 (fool)”判别器，使生成的假样本被判别器认为是真的。
        -   **Minimax 损失函数 (Minimax Loss Function)**:
            -   $\min_G \max_D V(G, D) = E_{x \sim p_{data}}[\log D(x)] + E_{z \sim p(z)}[\log (1 - D(G(z)))]$
            -   判别器 $D$ 试图最大化 $V(G,D)$：对真实数据 $x$ 输出 $D(x) \approx 1$，对生成数据 $G(z)$ 输出 $D(G(z)) \approx 0$。
            -   生成器 $G$ 试图最小化 $V(G,D)$：使 $D(G(z)) \approx 1$（欺骗判别器）。
    -   **训练过程 (Training Procedure)**:
        -   使用交替梯度更新 (alternating gradient updates) 联合训练 $G$ 和 $D$。
        -   更新 $D$: 梯度上升 (gradient ascent) $\alpha_D \frac{\partial V}{\partial D}$ (最大化 $D$ 的性能)。
        -   更新 $G$: 梯度下降 (gradient descent) $\alpha_G \frac{\partial V}{\partial G}$ (最小化 $G$ 的损失)。
    -   **原版 GAN 的问题 (Problems with Original GANs)**:
        1.  **没有整体损失最小化 (No overall loss minimization)**: 损失函数不是一个简单的下坡优化问题，难以判断训练进度和收敛。
        2.  **生成器梯度消失 (Vanishing gradients for G)**: 在训练初期，生成器很差，判别器很容易区分真假。此时 $D(G(z))$ 接近 0，导致 $\log(1 - D(G(z)))$ 饱和，生成器的梯度很小，学习停滞。
            -   **解决方案 (Solution)**: 在实践中，通常将生成器的损失函数改为最大化 $\log(D(G(z)))$。这样在判别器很强时，生成器也能获得较大的梯度信号。

-   **生成对抗网络：最优性 (GANs: Optimality)**
    -   **理论证明 (Theoretical Proof)**: 在某些假设下，Minimax 博弈的全局最优解 (global optimum) 发生在 $p_G = p_{data}$ 时。
    -   数学推导 (Mathematical Derivation):
        -   Minimax 目标: $\min_G \max_D (E_{x \sim p_{data}}[\log D(x)] + E_{z \sim p(z)}[\log (1 - D(G(z)))])$
        -   将 $E_{z \sim p(z)}[\log (1 - D(G(z)))]$ 项的 $G(z)$ 转换为 $x$ 对应的分布 $p_G(x)$：
            $\min_G \max_D \int_x (p_{data}(x) \log D(x) + p_G(x) \log (1 - D(x))) dx$
        -   对于任意固定的 $G$，判别器的最优解 $D^*(x)$ 可通过求解 $\frac{\partial}{\partial D(x)} (p_{data}(x) \log D(x) + p_G(x) \log (1 - D(x))) = 0$ 得到。
        -   最优判别器 (Optimal Discriminator): $D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}$
        -   将 $D_G^*(x)$ 代回原始目标函数，经过一系列数学变换和化简，最终目标函数可以表示为：
            $\min_G (2 \cdot JSD(p_{data}, p_G) - \log 4)$
            其中 $JSD(p_{data}, p_G)$ 是真实数据分布和生成器分布之间的 Jensen-Shannon Divergence。
        -   **Jensen-Shannon Divergence (JSD)**: $JSD(p, q) = \frac{1}{2} KL(p, \frac{p+q}{2}) + \frac{1}{2} KL(q, \frac{p+q}{2})$
        -   JSD 具有非负性，且当且仅当 $p=q$ 时，JSD 等于零。
        -   因此，当 $JSD(p_{data}, p_G) = 0$ 时，目标函数达到全局最小值。这发生在 $p_G = p_{data}$ 的时候。
    -   **总结 (Summary)**:
        1.  最优判别器 (Optimal discriminator for any G): $D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}$
        2.  最优生成器 (Optimal generator for optimal D): $p_G(x) = p_{data}(x)$

    -   **注意事项 (Caveats)**:
        1.  理论证明假设生成器 $G$ 和判别器 $D$ 具有无限的表达能力 (expressive capacity)，能够表示任何函数。但在实际中，它们是具有固定架构的神经网络，不一定能完全代表最优的 $D^*$ 和 $G^*$。
        2.  理论证明未说明训练过程的收敛性 (convergence)。实际训练中可能存在模式崩溃 (mode collapse) 或训练不稳定等问题。

-   **生成对抗网络：结果 (GANs: Results)**
    -   **早期 GANs (Early GANs) (2014)**:
        -   能生成手写数字和模糊的人脸，虽然质量不高，但能识别出生成的新样本与训练集中最近邻样本不同，表明模型在生成新数据而非简单记忆。
    -   **DCGAN (Deep Convolutional GAN) (2016)**:
        -   使用深度卷积网络，显著提高了生成图像的质量和稳定性。
        -   能生成看起来“合理”的卧室图像，展示了对场景结构的理解。
        -   **潜在空间插值 (Interpolation in Latent Space)**: 在潜在空间中进行线性插值，生成的图像能平滑地从一个卧室过渡到另一个卧室，表明潜在空间具有语义连续性。
        -   **潜在空间向量算术 (Vector Math in Latent Space)**:
            -   通过对不同属性的图像（如“微笑的女人”、“中性的女人”、“中性的男人”）的潜在向量进行平均并进行加减运算，可以得到具有新组合属性的图像（如“微笑的男人”）。
            -   例如: 平均(带眼镜的男人) - 平均(不带眼镜的男人) + 平均(不带眼镜的女人) = 平均(带眼镜的女人)。
            -   这表明潜在空间能够捕获并解耦语义概念。
    -   **GAN 的爆炸式增长 (Explosion of GANs) (2017至今)**:
        -   相关研究论文数量呈指数级增长，不断涌现新的 GAN 变体和应用。
    -   **GAN 改进：改进的损失函数 (Improved Loss Functions)**:
        -   **Wasserstein GAN (WGAN)** (2017): 使用 Wasserstein 距离作为损失函数，解决了原始 GAN 的训练不稳定和模式崩溃问题，生成的样本质量有所提高。
        -   **WGAN with Gradient Penalty (WGAN-GP)** (2017): 在 WGAN 基础上引入梯度惩罚，进一步稳定训练并提高生成质量。
    -   **GAN 改进：更高分辨率 (Higher Resolution)**:
        -   **Progressive GAN (PGGAN)** (2018): 通过渐进式训练（从低分辨率到高分辨率逐渐添加网络层）生成高分辨率（1024x1024）的逼真人脸图像。
        -   **StyleGAN (2019)**: 基于 PGGAN 架构的改进，通过引入样式控制 (style control) 和噪声注入 (noise injection) 机制，生成极度逼真的、可控的人脸，甚至能用于图像到图像的转换。
    -   **GANs 不仅用于图像 (GANs: Not just for images!)**:
        -   **视频生成 (Generating Videos)**: 初步研究能生成短视频片段，但分辨率和帧率仍受限。
        -   **条件生成：不只是标签 (Conditioning on more than labels!)**:
            -   **文本到图像 (Text to Image)**:
                -   StackGAN/StackGAN++ (2016/2018): 根据文本描述生成图像。
                -   输入文本描述 (Y)，输出对应图像 (X)。
            -   **图像超分辨率 (Image Super-Resolution)**:
                -   SRGAN (2017): 输入低分辨率图像 (Y)，输出高分辨率图像 (X)。
            -   **图像到图像翻译 (Image-to-Image Translation)**:
                -   Pix2Pix (2017): 输入一种图像表示（如语义分割图、航拍图、白天照片、草图），输出另一种图像表示（如真实街景、地图、夜晚照片、真实物体）。
                -   CycleGAN (2017): 无需成对数据 (unpaired data) 即可进行图像到图像的翻译（如莫奈画风转换为照片，斑马转换为马，夏季转换为冬季，甚至视频转换）。
        -   **GANs：非图像数据 (GANs: Non-Image Data)**:
            -   **轨迹预测 (Trajectory Prediction)**: Social GAN (2019) 可用于预测人群未来的行走轨迹，考虑到社会可接受性。

### 二、关键术语定义 (Key Term Definitions)

-   **监督学习 (Supervised Learning)**: 模型从带有标签的训练数据中学习输入到输出的映射。
-   **无监督学习 (Unsupervised Learning)**: 模型从无标签数据中学习数据的内在结构或模式。
-   **判别模型 (Discriminative Model)**: 直接学习输入到输出的条件概率 $p(y|x)$ 的模型。
-   **生成模型 (Generative Model)**: 学习数据本身分布 $p(x)$ 或输入与输出的联合分布 $p(x, y)$ 的模型，可以用于生成新数据。
-   **条件生成模型 (Conditional Generative Model)**: 学习在给定某些条件 $y$ 的情况下生成数据 $x$ 的模型，即 $p(x|y)$。
-   **自回归模型 (Autoregressive Models)**: 通过将联合概率分解为一系列条件概率的乘积来建模数据，通常按顺序生成数据。
-   **像素循环神经网络 (PixelRNN)**: 使用 RNN 生成图像像素，每个像素依赖之前生成的像素。
-   **像素卷积神经网络 (PixelCNN)**: 使用 CNN 生成图像像素，通过掩码卷积确保每个像素只依赖“之前”的像素。
-   **变分自编码器 (Variational Autoencoder - VAE)**: 结合自编码器和概率图模型的生成模型，通过学习数据的潜在表示来生成数据，最大化似然的变分下限。
-   **潜在变量 (Latent Variable) / 隐变量**: 模型中未被直接观测到的变量，捕获数据的底层结构。
-   **编码器 (Encoder)**: VAE 的一部分，将输入数据映射到潜在空间的概率分布。
-   **解码器 (Decoder)**: VAE 的一部分，从潜在空间中的样本生成（或重构）数据。
-   **变分下限 (Variational Lower Bound - VLB)**: 用于近似优化数据对数似然的一个可计算的下界。
-   **KL散度 (Kullback-Leibler Divergence - $D_{KL}$)**: 衡量两个概率分布之间差异的指标。
-   **对角高斯分布 (Diagonal Gaussian Distribution)**: 协方差矩阵只有对角线元素的多元高斯分布，维度独立。
-   **解缠变异因子 (Disentangling factors of variation)**: 学习到的潜在变量维度对应数据独立可解释的生成因素。
-   **VQ-VAE2 (Vector-Quantized Variational Autoencoder 2)**: 结合 VAE 和自回归模型的生成模型，通过量化潜在空间和多尺度 PixelCNN 提高生成质量。
-   **生成对抗网络 (Generative Adversarial Network - GAN)**: 由一个生成器和一个判别器组成的生成模型，通过两者之间的对抗训练来学习数据分布。
-   **生成器 (Generator)**: GAN 的一部分，负责从随机噪声中生成假数据样本。
-   **判别器 (Discriminator)**: GAN 的一部分，负责判断输入的样本是真实的还是由生成器生成的。
-   **Minimax 博弈 (Minimax Game)**: GAN 训练中的对抗过程，生成器试图最小化判别器区分真假的能力，判别器试图最大化这种能力。
-   **梯度消失 (Vanishing Gradients)**: 神经网络训练中的一个问题，指梯度变得非常小，导致模型参数更新缓慢或停滞。
-   **Wasserstein GAN (WGAN)**: 一种 GAN 变体，使用 Wasserstein 距离作为损失函数，提高了训练的稳定性和生成样本的质量。
-   **Jensen-Shannon Divergence (JSD)**: 衡量两个概率分布之间相似性的对称指标，是 KL 散度的对称和。
-   **DCGAN (Deep Convolutional GAN)**: 一种使用深度卷积网络的 GAN 架构，显著提高了图像生成质量。
-   **Progressive GAN (PGGAN)**: 通过渐进式训练（从低分辨率到高分辨率）生成高分辨率图像的 GAN。
-   **StyleGAN**: PGGAN 的改进，引入样式控制和噪声注入，生成更逼真和可控的图像。
-   **图像超分辨率 (Image Super-Resolution)**: 从低分辨率图像生成高分辨率图像的任务。
-   **图像到图像翻译 (Image-to-Image Translation)**: 将图像从一个领域转换为另一个领域，保持内容不变但改变风格或属性。
-   **CycleGAN**: 一种无需成对训练数据即可进行图像到图像翻译的 GAN。
-   **谱范数归一化 (Spectral Normalization)**: GAN 训练中用于稳定判别器的一个正则化技术。
-   **自注意力 (Self-Attention)**: 一种神经网络机制，允许模型在处理序列或图像数据时，对输入的不同部分分配不同的权重。
-   **BigGAN**: 目前最大、最先进的条件 GAN 模型之一，能够生成高度逼真的大尺寸图像。
-   **轨迹预测 (Trajectory Prediction)**: 预测物体或人群未来运动路径的任务。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **自回归模型的密度函数 (Density Function of Autoregressive Models)**:
    -   $p(x) = p(x_1, x_2, ..., x_T)$
    -   $= p(x_1)p(x_2 | x_1)p(x_3 | x_1, x_2) \dots$
    -   $= \prod_{t=1}^{T} p(x_t | x_1, \dots, x_{t-1})$

-   **变分自编码器 (Variational Autoencoders - VAEs) 训练目标**:
    -   联合训练编码器 $q$ 和解码器 $p$ 以最大化数据似然的变分下限。
    -   $\log p_{\theta}(x) \ge E_{Z \sim q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x), p(z))$
    -   **KL散度 (Kullback-Leibler Divergence) 的闭式解 (Closed-form Solution)** (当 $q_{\phi}(z|x)$ 为对角高斯且 $p(z)$ 为单位高斯时):
        $D_{KL}(N(\mu, \Sigma), N(0, I)) = \frac{1}{2} \sum_{j=1}^{J} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$
        其中 $\mu_j$ 和 $\sigma_j^2$ 分别是潜在变量 $z$ 第 $j$ 维的均值和方差。

-   **生成对抗网络 (Generative Adversarial Networks - GANs) 训练目标**:
    -   $\min_G \max_D V(G, D) = E_{x \sim p_{data}}[\log D(x)] + E_{z \sim p(z)}[\log (1 - D(G(z)))]$
    -   **生成器梯度消失问题的解决方案**: 训练 $G$ 最大化 $E_{z \sim p(z)}[\log D(G(z))]$ 而不是最小化 $E_{z \sim p(z)}[\log (1 - D(G(z)))]$。

-   **GAN 最优性数学推导过程**:
    -   **转换为积分形式**: $\min_G \max_D \int_x (p_{data}(x) \log D(x) + p_G(x) \log (1 - D(x))) dx$
    -   **求解最优判别器 $D_G^*(x)$**:
        $D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}$
    -   **代入 $D_G^*(x)$ 并化简目标函数**:
        $\min_G (2 \cdot JSD(p_{data}, p_G) - \log 4)$
        其中 $JSD(p_{data}, p_G)$ 是 Jensen-Shannon Divergence。
    -   **Jensen-Shannon Divergence (JSD) 定义**:
        $JSD(p, q) = \frac{1}{2} KL(p, \frac{p+q}{2}) + \frac{1}{2} KL(q, \frac{p+q}{2})$
    -   **JSD 特性**: 始终非负 ($JSD \ge 0$)，且仅当 $p=q$ 时 $JSD=0$。
    -   **全局最小值**: 当 $p_G = p_{data}$ 时，JSD 达到 0，目标函数取得全局最小值。

-   **条件 Batch Normalization (Conditional Batch Normalization)**:
    -   标准 Batch Normalization: $\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$，然后 $y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j$。
    -   条件 Batch Normalization: 针对每个不同的标签 $y$ 学习不同的缩放参数 $\gamma_j^y$ 和平移参数 $\beta_j^y$。
        $y_{i,j} = \gamma_j^y \hat{x}_{i,j} + \beta_j^y$。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)

-   Is a dog more likely to sit or stand? How about a 3-legged dog vs 3-armed monkey? (关于生成模型对数据“合理性”的理解)
-   Should we assume different priors for different data sets? (关于 VAE 中先验分布的选择)
-   Can we combine autoregressive models and variational models and get the best of both worlds? (引出 VQ-VAE2 的动机)

---
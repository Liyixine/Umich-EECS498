### [📚] 视频学习脚手架: Lecture 3: Linear Classifiers

### 一、核心内容大纲 (Core Content Outline)
-   **讲座介绍 (Lecture Introduction)**
    -   主题：线性分类器 (Linear Classifiers)
-   **上节回顾：图像分类 (Last Time: Image Classification Recap)**
    -   **图像分类问题 (Image Classification Problem)**
        -   输入：图像 (Input: image)
        -   输出：将图像分配到固定类别的其中一个 (Output: Assign image to one of a fixed set of categories)
        -   基础问题 (Foundational Problem)
    -   **识别的挑战 (Challenges of Recognition)**
        -   视点变化 (Viewpoint changes)
        -   光照变化 (Illumination changes)
        -   形变 (Deformation)
        -   遮挡 (Occlusion)
        -   杂乱 (Clutter)
        -   类内差异 (Intraclass Variation)
        -   需要构建对这些变化鲁棒的分类器 (Need classifiers robust to variations)
    -   **数据驱动方法：kNN (Data-Driven Approach: kNN)**
        -   不尝试写出显式函数来处理所有视觉细节 (Rather than explicit function for visual details)
        -   收集大量数据 (Collect a big dataset)
        -   使用学习算法从数据中学习 (Use a learning algorithm to learn from data)
        -   kNN 分类器 (kNN classifier)
            -   记忆训练数据 (Memorize training data)
            -   测试时输出最相似训练图像的标签 (Output label of most similar training image at test time)
        -   kNN的缺点 (Limitations of kNN)
            -   训练快，评估慢 (Fast at training, slow to evaluate)
            -   L1/L2距离在像素值上不具有感知意义 (L1/L2 distances on raw pixel values not perceptually meaningful)
-   **今日主题：线性分类器 (Today: Linear Classifiers)**
    -   重要性：作为神经网络的基本构建模块 (Very important as basic blocks in neural networks)
    -   **回顾 CIFAR-10 数据集 (Recall CIFAR-10 Dataset)**
        -   50,000张训练图像 (50,000 training images)
        -   10,000张测试图像 (10,000 test images)
        -   每张图像尺寸：32x32x3 (Each image is 32x32x3 numbers)
            -   共3072个像素值 (3072 numbers total)
    -   **参数化方法 (Parametric Approach)**
        -   核心思想：一个函数 $f(x,W)$，输入图像 $x$ 和可学习的权重 $W$ (learnable weights)，输出分类得分 (class scores)。
        -   输出为10个数字，对应10个类别得分 (10 numbers giving class scores)
        -   **线性分类器公式 (Linear Classifier Formula)**: $f(x,W) = Wx + b$
            -   $x$：输入图像的像素值展平为列向量 (Input image pixels stretched into a column vector, e.g., 3072 dimensions for CIFAR-10).
            -   $W$：权重矩阵 (Weight matrix)
                -   形状：(类别数, 像素维度) (Shape: e.g., (10, 3072) for CIFAR-10)
            -   $b$：偏置向量 (Bias vector)
                -   形状：(类别数,) (Shape: e.g., (10,) for CIFAR-10)
            -   输出：分类得分向量 (Output: Vector of scores, e.g., (10,) for CIFAR-10)
        -   **偏置技巧 (Bias Trick)**
            -   将偏置 $b$ 吸收到权重矩阵 $W$ 中 (Bias absorbed into weight matrix $W$)
            -   通过在输入向量 $x$ 的末尾添加一个常数1 (Add extra one to data vector)
            -   相应地增加 $W$ 的列数 (Increase last column of weight matrix)
    -   **解释线性分类器 (Interpreting a Linear Classifier)**
        -   **代数视角 (Algebraic Viewpoint)**: $f(x,W) = Wx + b$
            -   将图像像素展平为列向量进行矩阵向量乘法 (Stretch pixels into column for matrix-vector multiplication).
        -   **视觉视角 (Visual Viewpoint)**: 线性分类器有一个“模板”每个类别 (Linear classifier has one "template" per category)
            -   权重矩阵 $W$ 的每一行可以被重新解释为对应类别的图像模板 (Each row of $W$ can be reshaped to match image dimensions).
            -   分类得分是通过输入图像和每个类别模板的内积（匹配度）计算的 (Scores are inner products/matches between input image and class templates).
            -   **局限性 (Limitations)**:
                -   单个模板无法捕捉数据中的多种模式 (A single template cannot capture multiple modes of the data).
                -   例如：马可能朝向不同方向，导致模板融合了左右朝向的特征，看起来像有两只头 (e.g., horse template has 2 heads due to averaging different poses).
                -   图像的背景或上下文信息会强烈影响分类 (Relies heavily on context cues, e.g., blue sky for airplane, green background for deer).
                -   对颜色等简单缩放敏感 (Predictions are linear, scaling image pixels by c scales scores by c, which is unintuitive for humans).
        -   **几何视角 (Geometric Viewpoint)**: 超平面切割高维空间 (Hyperplanes carving up a high-dimensional space)
            -   每个类别的分类得分函数在像素空间中是线性的 (Classifier score is a linear function of pixel values).
            -   得分为零的像素形成一个超平面 (The set of pixels where the score is zero forms a hyperplane).
            -   每个类别由一个超平面定义 (Each class is defined by a hyperplane).
            -   **局限性 (Limitations)**:
                -   几何直觉在高维空间中可能失效 (Geometry gets really weird in high dimensions).
                -   线性分类器无法处理非线性可分的数据 (Cannot learn XOR function, concentric circles, or classes with multiple disjoint modes).
                -   这是感知器 (Perceptron) 失败的原因之一 (Historical context: Perceptron couldn't learn XOR because it's a linear classifier).
-   **选择好的权重 $W$ (Choosing a good W)**
    -   目前为止，我们只定义了一个线性得分函数 $f(x,W) = Wx + b$ (So far: Defined a linear score function).
    -   但如何选择一个好的 $W$ 呢？ (But how can we actually choose a good W?)
    -   **待办事项 (TODO)**:
        1.  使用一个损失函数 (Loss Function) 来量化当前 $W$ 的好坏 (Use a loss function to quantify how good a value of W is).
            -   低损失 = 好的分类器 (Low loss = good classifier)
            -   高损失 = 坏的分类器 (High loss = bad classifier)
            -   也称为：目标函数 (objective function)、成本函数 (cost function)
            -   负损失函数有时称为奖励函数 (reward function)、利润函数 (profit function)、效用函数 (utility function)、适应度函数 (fitness function) 等 (Negative loss function sometimes called reward function, profit function, utility function, fitness function, etc.).
        2.  找到一个最小化损失函数 $W$ (优化) (Find a W that minimizes the loss function (optimization)).
    -   **损失函数定义 (Loss Function Definition)**:
        -   给定一个示例数据集 $\{(x_i, y_i)\}_{i=1}^N$ (Given a dataset of examples).
            -   $x_i$ 是图像 (where $x_i$ is image)
            -   $y_i$ 是整数标签 (and $y_i$ is integer label)
        -   单个示例的损失 $L_i = L_i(f(x_i, W), y_i)$ (Loss for a single example is $L_i$).
        -   整个数据集的损失是每个示例损失的平均值 $L = \frac{1}{N} \sum_{i} L_i(f(x_i, W), y_i)$ (Loss for the dataset is average of per-example losses).
    -   **多分类支持向量机损失 (Multiclass SVM Loss)**
        -   核心思想：正确类别的得分应该高于所有其他类别的得分 (The score of the correct class should be higher than all the other scores).
        -   **公式 (Formula)**: $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$
            -   $s = f(x_i, W)$ 是模型输出的得分向量 ($s$ are scores).
            -   $s_j$ 是错误类别的得分 ($s_j$ is score for incorrect class).
            -   $s_{y_i}$ 是正确类别的得分 ($s_{y_i}$ is score for correct class).
            -   $+1$ 是一个“边界”或“间隔” (a "margin").
        -   **特性 (Properties)**:
            -   当正确类别的得分显著高于最高不正确类别的得分（超过1的间隔）时，损失为0 (If correct score is sufficiently higher than others, loss is 0).
            -   否则，损失线性增加 (Otherwise, loss increases linearly).
            -   一旦正确分类，微小的分数变化不会影响损失 (If correctly classified, small score changes don't affect loss).
            -   最小可能损失为0 (Min possible loss = 0).
            -   最大可能损失无上限 (Max possible loss is unbounded).
    -   **正则化 (Regularization)**
        -   目的：超越训练误差，防止模型在训练数据上表现“过好” (Prevent the model from doing *too* well on training data).
        -   **全损失函数 (Full Loss Function)**: $L(W) = \frac{1}{N} \sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda R(W)$
            -   `数据损失 (Data loss)`: 模型预测应与训练数据匹配 (Model predictions should match training data).
            -   `正则化项 (Regularization term)`: 惩罚复杂模型 (Penalizes complex models).
            -   $\lambda$: 正则化强度 (regularization strength)，一个超参数 (hyperparameter)。
        -   **简单正则化示例 (Simple Regularization Examples for Linear Models)**:
            -   `L2正则化 (L2 regularization)`: $R(W) = \sum_k \sum_l W_{k,l}^2$ (倾向于“平摊”权重，使用所有特征).
            -   `L1正则化 (L1 regularization)`: $R(W) = \sum_k \sum_l |W_{k,l}|$ (倾向于产生稀疏权重，只用少量特征).
            -   `弹性网络 (Elastic Net)`: $R(W) = \sum_k \sum_l \beta W_{k,l}^2 + |\alpha W_{k,l}|$ (L1 + L2组合).
        -   **更复杂正则化示例 (More Complex Regularization for Neural Networks)**:
            -   `Dropout` (随机丢弃神经元).
            -   `批标准化 (Batch Normalization)` (标准化层输入).
            -   `Cutout`, `Mixup`, `Stochastic depth` (数据增强及网络结构随机化方法).
        -   **正则化的目的 (Purpose of Regularization)**:
            -   在“最小化训练误差”之外，表达模型间的偏好 (Express preferences in among models beyond "minimize training error").
            -   避免**过拟合 (Overfitting)**: 偏好简单模型，使其泛化能力更好 (Prefer simple models that generalize better).
            -   通过增加曲率改进优化过程 (Improve optimization by adding curvature)。
    -   **交叉熵损失 (Cross-Entropy Loss)**
        -   目的：将原始分类器得分解释为**概率 (probabilities)** (Want to interpret raw classifier scores as probabilities).
        -   **Softmax函数 (Softmax Function)**: $P(Y=k|X=x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$
            -   将得分 $s$ 转换为非负的非归一化概率 (Take raw scores, apply exponential to make them non-negative).
            -   然后进行归一化，使其总和为1 (Normalize by dividing by the sum of exponentials).
            -   结果是一组可以解释为类别的概率 (Result is a probability distribution over classes).
        -   **交叉熵损失公式 (Cross-Entropy Loss Formula)**: $L_i = -\log P(Y=y_i|X=x_i)$
            -   最大化正确类别的概率 (Maximizes probability of correct class).
            -   是**最大似然估计 (Maximum Likelihood Estimation)** 的一个实例 (It's an instance of MLE).
        -   **与 Kullback-Leibler (KL) 散度的关系 (Relation to KL Divergence)**:
            -   $L_i$ 等价于计算预测概率分布和真实概率分布之间的 KL 散度 (The loss can be seen as minimizing the KL divergence between predicted and true distributions).
            -   $D_{KL}(P||Q) = \sum_y P(y) \log \frac{P(y)}{Q(y)}$ (Kullback-Leibler divergence formula).
            -   交叉熵 (Cross Entropy) 也是信息论中的一个概念：$H(P, Q) = H(P) + D_{KL}(P||Q)$。
-   **比较交叉熵损失与SVM损失 (Cross-Entropy Loss vs SVM Loss Comparison)**
    -   **SVM 损失 (SVM Loss)**：一旦达到边距，损失就为0，不再关心得分的进一步增加 (Once the margin is met, loss is 0 and no further improvement is penalized).
    -   **交叉熵损失 (Cross-Entropy Loss)**：永不满足，总是希望正确类别的得分无限高 (Never satisfied, always wants the score of the correct class to be infinitely higher).

### 二、关键术语定义 (Key Term Definitions)
-   **线性分类器 (Linear Classifiers)**: 一种机器学习模型，通过输入特征的线性组合来做出分类决策，其决策边界是一个超平面。
-   **图像分类 (Image Classification)**: 计算机视觉任务，旨在将图像分配到预定义的类别之一。
-   **数据驱动方法 (Data-Driven Approach)**: 一种机器学习范式，通过从大量数据中学习模式来构建模型，而不是通过人工编写规则。
-   **K最近邻 (k-Nearest Neighbors, kNN)**: 一种简单的非参数分类算法，通过查找训练数据集中与新数据点最近的k个样本来进行分类。
-   **超参数 (Hyperparameters)**: 在模型训练开始前需要手动设定的参数，例如kNN中的k值。
-   **交叉验证 (Cross-Validation)**: 一种模型验证技术，用于评估模型在独立数据集上的泛化能力，通过将数据集分成多个子集进行训练和测试。
-   **感知意义 (Perceptually Meaningful)**: 指的是某种度量或特征与人类感知的相似或相关程度。
-   **参数化方法 (Parametric Approach)**: 一种机器学习方法，其中模型具有固定数量的参数（如权重和偏置），这些参数从数据中学习。
-   **权重 (Weights)**: 参数化模型中的可学习数值，它们决定了输入特征对模型输出的影响。
-   **偏置 (Bias)**: 参数化模型中的可学习数值，作为常数项添加到线性组合中，允许决策边界不经过原点。
-   **CIFAR-10 数据集 (CIFAR-10 Dataset)**: 一个常用的计算机视觉数据集，包含10个类别的60,000张32x32彩色图像。
-   **偏置技巧 (Bias Trick)**: 一种简化线性模型表示的方法，通过在输入特征向量中添加一个值为1的额外维度，将偏置项吸收到权重矩阵中。
-   **代数视角 (Algebraic Viewpoint)**: 从数学代数（如矩阵乘法）的角度理解和表示线性分类器的工作原理。
-   **视觉视角 (Visual Viewpoint)**: 从视觉模式或“模板”匹配的角度理解线性分类器的工作原理，每个类别对应一个 learned template。
-   **模板匹配 (Template Matching)**: 一种图像处理技术，通过比较输入图像与预定义模板的相似性来识别对象或模式。
-   **几何视角 (Geometric Viewpoint)**: 从几何空间（如像素空间）的角度理解线性分类器，其中决策边界被表示为超平面。
-   **超平面 (Hyperplanes)**: N维空间中维度为N-1的平坦子空间，在线性分类器中作为不同类别之间的决策边界。
-   **损失函数 (Loss Function)**: 一个数学函数，用于量化模型预测与真实标签之间的不匹配程度，目标是最小化此函数。
-   **目标函数 (Objective Function)**: 通常与损失函数同义，表示模型优化所追求的目标。
-   **成本函数 (Cost Function)**: 损失函数或目标函数的另一个同义词，通常指在整个训练数据集上的平均损失。
-   **多分类支持向量机损失 (Multiclass SVM Loss)**: 一种常用的损失函数，它鼓励正确类别的得分比所有不正确类别的得分至少高出一个预设的间隔。
-   **铰链损失 (Hinge Loss)**: 多分类SVM损失的具体形式，其图形表示为一个“铰链”形状，即在达到一定间隔后损失变为零。
-   **边际/间隔 (Margin)**: 在SVM损失中，正确分类的得分需要超过不正确分类的得分的最小差值。
-   **数据损失 (Data Loss)**: 损失函数的一部分，量化模型预测与训练数据之间的匹配程度。
-   **正则化 (Regularization)**: 在机器学习中，通过向损失函数添加一个正则化项来防止模型过拟合，鼓励模型学习更简单、泛化能力更好的参数。
-   **L2正则化 (L2 Regularization)**: 一种正则化技术，将模型权重的平方和添加到损失函数中，惩罚大的权重值，促使权重分散且较小。
-   **L1正则化 (L1 Regularization)**: 一种正则化技术，将模型权重的绝对值和添加到损失函数中，倾向于产生稀疏模型（即许多权重为零），用于特征选择。
-   **弹性网络 (Elastic Net)**: 结合了L1和L2正则化的一种方法。
-   **Dropout (随机失活)**: 一种神经网络正则化技术，在训练过程中随机地丢弃（置零）一部分神经元的输出，以防止过拟合。
-   **批标准化 (Batch Normalization)**: 一种神经网络技术，用于标准化网络层输入，从而稳定和加速训练过程，并具有正则化效果。
-   **过拟合 (Overfitting)**: 指模型在训练数据上表现良好，但在未见过的新数据上表现不佳的现象。
-   **交叉熵损失 (Cross-Entropy Loss)**: 一种常用的损失函数，特别适用于分类问题，它衡量了两个概率分布之间的差异。
-   **未归一化对数概率/Logits (Unnormalized Log-probabilities / Logits)**: 在Softmax函数应用之前，线性分类器输出的原始得分。
-   **Softmax函数 (Softmax Function)**: 一种将任意实值向量转换为概率分布的函数，其输出值在0到1之间且总和为1。
-   **最大似然估计 (Maximum Likelihood Estimation, MLE)**: 一种统计方法，用于估计模型参数，使得观测数据的概率最大化。
-   **Kullback-Leibler (KL) 散度 (Kullback-Leibler (KL) Divergence)**: 衡量两个概率分布之间差异的非对称度量，在信息论和机器学习中常用于比较模型预测分布与真实分布。
-   **交叉熵 (Cross Entropy)**: 在信息论中，交叉熵衡量了使用一个编码方案来表示另一个编码方案所需要的平均比特数，在分类任务中，它被用作衡量预测概率分布与真实分布之间差异的损失函数。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **线性分类器得分函数 (Linear Classifier Score Function)**:
    -   $f(x,W) = Wx$ (when bias trick is applied where x includes a constant 1 at the end).

-   **多分类支持向量机损失 (Multiclass SVM Loss)**:
    -   **公式 (Formula)**: $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$
    -   **单个示例的SVM损失计算示例 (SVM Loss Calculation Example for a Single Example)**:
        -   **图像** (Image): 猫 (Cat), **真实标签** (True Label): cat (对应索引 0)
        -   **模型预测得分** (Model Predicted Scores): $s = [3.2, 5.1, -1.7]$ (cat: 3.2, car: 5.1, frog: -1.7)
        -   **损失计算** (Loss Calculation):
            $L_{cat} = \max(0, s_{car} - s_{cat} + 1) + \max(0, s_{frog} - s_{cat} + 1)$
            $L_{cat} = \max(0, 5.1 - 3.2 + 1) + \max(0, -1.7 - 3.2 + 1)$
            $L_{cat} = \max(0, 1.9 + 1) + \max(0, -4.9 + 1)$
            $L_{cat} = \max(0, 2.9) + \max(0, -3.9)$
            $L_{cat} = 2.9 + 0$
            $L_{cat} = 2.9$
        -   **平均损失 (Average Loss)** (基于视频中给出的所有三个图像的损失): $L = (2.9 + 0 + 12.9) / 3 = 5.27$

-   **交叉熵损失 (Cross-Entropy Loss)**:
    -   **Softmax函数 (Softmax Function)**: $P(Y=k|X=x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$
    -   **交叉熵损失公式 (Cross-Entropy Loss Formula)**: $L_i = -\log P(Y=y_i|X=x_i)$ (Put it all together: $L_i = -\log \left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right)$)
    -   **单个示例的交叉熵损失计算示例 (Cross-Entropy Loss Calculation Example for a Single Example)**:
        -   **图像** (Image): 猫 (Cat), **真实标签** (True Label): cat (对应索引 0)
        -   **模型预测得分** (Model Predicted Scores): $s = [3.2, 5.1, -1.7]$
        -   **步骤1: 指数化 (Exponentiate)**:
            $e^{3.2} \approx 24.5$
            $e^{5.1} \approx 164.0$
            $e^{-1.7} \approx 0.18$
            (这些是未归一化概率 (unnormalized probabilities)，必须大于等于0 (must be $\geq 0$))
        -   **步骤2: 归一化 (Normalize)** (通过Softmax函数):
            总和 $= 24.5 + 164.0 + 0.18 = 188.68$
            概率 (probabilities):
            猫 (Cat): $24.5 / 188.68 \approx 0.13$
            车 (Car): $164.0 / 188.68 \approx 0.87$
            青蛙 (Frog): $0.18 / 188.68 \approx 0.00$
            (这些概率总和必须为1 (must sum to 1))
        -   **步骤3: 计算损失 (Calculate Loss)**:
            $L_{cat} = -\log P(Y=cat|X=x_{cat})$
            $L_{cat} = -\log(0.13)$
            $L_{cat} \approx 2.04$

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   **Q1: "What happens to the loss if the scores for the car image change a bit?"** (如果汽车图像的得分发生一点变化，损失会怎样？)
    -   **对于SVM损失 (For SVM Loss)**: 损失将保持不变。因为汽车图像已经正确分类且满足了1的边际，损失已经为0。微小的分数变化不会改变这个状态。
-   **Q2: "What are the min and max possible loss?"** (最小和最大可能损失是多少？)
    -   **对于SVM损失 (For SVM Loss)**: 最小损失为0。最大损失是无限大。
    -   **对于交叉熵损失 (For Cross-Entropy Loss)**: 最小损失为0（当模型对正确类别分配100%的概率时）。最大损失是无限大（当模型对正确类别分配0%的概率时）。
-   **Q3: "If all the scores were random, what loss would we expect?"** (如果所有得分都是随机的小值，期望的损失是多少？)
    -   **对于SVM损失 (For SVM Loss)**: 期望的损失约为 $C-1$，其中 $C$ 是类别数。例如，对于3个类别，期望损失为2。
    -   **对于交叉熵损失 (For Cross-Entropy Loss)**: 期望的损失约为 $-\log(1/C)$ 或 $\log(C)$。例如，对于10个类别，期望损失约为 $\log(10) \approx 2.3$。
    -   这是一个有用的调试技巧：如果你的模型在随机初始化后没有得到这个损失值，那么你的实现可能存在错误。
-   **Q4: "What would happen if the sum were over all classes? (including i = yi)"** (如果求和是针对所有类别（包括正确类别 $i=y_i$），会发生什么？)
    -   **对于SVM损失 (For SVM Loss)**: 损失会增加一个常数1。因为 $max(0, s_{y_i} - s_{y_i} + 1) = max(0, 1) = 1$。
    -   **对于交叉熵损失 (For Cross-Entropy Loss)**: 损失的相对偏好不会改变，因为只是整体增加了一个常数，它仍然会产生相同的分类器偏好。
-   **Q5: "What if the loss used a mean instead of a sum?"** (如果损失函数使用平均值而不是求和，会怎样？)
    -   **对于SVM损失和交叉熵损失 (For both SVM and Cross-Entropy Loss)**: 损失的数值会按比例缩小，但对权重矩阵的偏好（即哪个权重矩阵更好）不会改变，因为这只是一个单调变换。
-   **Q6: "What if we used this loss instead?" ($L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)^2$)** (如果使用这个损失函数会怎样？)
    -   **对于SVM损失 (For SVM Loss)**: 这将表达不同的权重矩阵偏好。因为它引入了平方项，会非线性地改变得分对损失的影响。它将不再被称为多分类SVM损失。
-   **Q7: (比较Cross-Entropy Loss和SVM Loss在特定得分下的行为) What happens to each loss if I slightly change the scores of the last datapoint?** (如果我稍微改变最后一个数据点的得分，每个损失会怎样？)
    -   **假设得分** (Assume scores): `[10, -100, -100]`，**正确类别** (Correct class) $y_i=0$ (得分为10)。
    -   **SVM损失 (SVM Loss)**: $max(0, -100-10+1) + max(0, -100-10+1) = 0+0 = 0$。如果稍微改变-100，损失仍然是0，因为它已经满足了边距。SVM损失对此不敏感。
    -   **交叉熵损失 (Cross-Entropy Loss)**: 损失会发生变化。交叉熵损失总是希望正确类别的概率尽可能高，即使它已经正确分类，也会继续推动得分分开。
-   **Q8: (比较Cross-Entropy Loss和SVM Loss在特定得分下的行为) What happens to each loss if I double the score of the correct class from 10 to 20?** (如果我将正确类别的得分从10加倍到20，每个损失会怎样？)
    -   **SVM损失 (SVM Loss)**: 损失仍然是0。因为已经满足了边距。
    -   **交叉熵损失 (Cross-Entropy Loss)**: 损失会**减小**。因为正确类别的概率会增加，$-\log(P)$ 就会减小。交叉熵损失总是会继续推动得分分开。
-   **Q9: (调试技巧) If all scores are small random values, what is the loss?** (如果所有得分都是小的随机值，损失是多少？)
    -   **对于交叉熵损失 (For Cross-Entropy Loss)**: 答案是 $-\log(C)$，其中 $C$ 是类别数。对于 CIFAR-10 ($C=10$)，这个值大约是 $\log(10) \approx 2.3$。
    -   这是一个重要的调试工具：如果你在训练开始时（权重随机初始化）没有看到接近这个值的损失，那么你的代码可能有bug。

---

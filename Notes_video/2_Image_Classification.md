### [📚] 视频学习脚手架: Lecture 2: Image Classification

### 一、核心内容大纲 (Core Content Outline)

-   **引言 (Introduction)**
    -   本次讲座为第二讲，主题是图像分类 (This is Lecture 2, on Image Classification)。
    -   回顾上次讲座内容：计算机视觉、深度学习和机器学习的历史概述 (Recap: Historical overview of computer vision, deep learning, and machine learning)。

-   **图像分类：核心计算机视觉任务 (Image Classification: A Core Computer Vision Task)**
    -   **定义 (Definition)**
        -   输入 (Input): 图像 (Image)。
        -   输出 (Output): 将图像分配到预定义的固定类别集合中的一个 (Assign image to one of a fixed set of categories)。
        -   示例类别 (Example categories): 猫 (cat), 鸟 (bird), 鹿 (deer), 狗 (dog), 卡车 (truck)。
    -   **人类感知与机器感知 (Human vs. Computer Perception)**
        -   人类 (Humans): 图像分类是微不足道的任务，几乎无需思考即可完成 (Trivial task, immediately know it's a cat without thinking)。
        -   计算机 (Computers): 远非易事 (Not so easy)。
            -   计算机看到的是什么？图像只是一个由0到255之间的数字组成的巨大网格 (What the computer sees: A big grid of numbers between 0 and 255)。
            -   例如：800 x 600 x 3 (3个RGB通道) (e.g., 800 x 600 x 3 (3 channels RGB))。
            -   没有明显的方法将原始像素值网格转换为具有语义意义的类别标签 (No obvious way to convert raw pixel values into semantically meaningful category labels)。

-   **图像分类的挑战 (Challenges in Image Classification)**
    -   **语义鸿沟 (Semantic Gap)**
        -   图像的微小变化可能导致像素值发生巨大变化 (Small changes in images can drastically change pixel values)。
    -   **视点变化 (Viewpoint Variation)**
        -   相机移动时，所有像素都会改变 (All pixels change when the camera moves)。
        -   算法需要对这些变化具有鲁棒性 (Algorithms need to be robust to these changes)。
    -   **类内差异 (Intraclass Variation)**
        -   同一类别内的不同对象（例如：不同的猫）看起来非常不同 (Different instances of the same category look very different)。
        -   算法需要对同一类别内可能发生的巨大变化具有鲁棒性 (Algorithms need to be robust to massive variations within categories)。
    -   **细粒度类别 (Fine-Grained Categories)**
        -   识别视觉上非常相似的不同子类别 (Recognizing different categories that appear very visually similar)。
        -   例如：识别不同品种的猫 (e.g., different breeds of cats like Maine Coon, Ragdoll, American Shorthair)。
    -   **背景杂乱 (Background Clutter)**
        -   图像中的物体可能与背景融合 (Objects in the image might blend into the background)。
        -   例如：由于自然伪装或其他场景中的复杂情况 (e.g., due to natural camouflage or other crazy things in the scene)。
    -   **光照变化 (Illumination Changes)**
        -   场景中的光照条件变化会导致像素值发生巨大变化 (Lighting conditions change significantly)。
        -   算法需要对不同光照条件下的巨大变化具有鲁棒性 (Algorithms should be robust to massive changes in different lighting conditions)。
    -   **形变 (Deformation)**
        -   物体可能以非常不同的姿态或位置出现 (Objects might appear in very different poses/positions)。
        -   例如：猫可以摆出各种姿势 (e.g., cats in various poses)。
    -   **遮挡 (Occlusion)**
        -   物体在图像中可能几乎不可见 (The object we want to recognize might not be visible hardly at all)。
        -   识别需要大量的常识性推理 (Recognition involves common-sense reasoning about the world, e.g., a tail sticking out from under a couch)。

-   **图像分类：非常有用！(Image Classification: Very Useful!)**
    -   **科学应用 (Scientific Applications)**
        -   医学成像 (Medical Imaging): 诊断良性/恶性肿瘤 (diagnosing benign/malignant tumors)。
        -   星系分类 (Galaxy Classification): 分类望远镜数据中的天体现象 (classifying celestial phenomena from telescope data)。
        -   鲸鱼识别 (Whale Recognition) 及其他动物分类 (and other animal classification)。
    -   **作为其他任务的基础模块 (Building Block for Other Tasks)**
        -   **目标检测 (Object Detection)**: 绘制边界框并分类图像中的物体 (Draw boxes around objects and classify them)。
        -   **图像标注 (Image Captioning)**: 给定输入图像，编写自然语言句子描述图像内容 (Given an input image, write a natural language sentence to describe what is in the image)。
        -   **玩围棋 (Playing Go)**: 输入是棋盘图像，输出是下一个落子的位置 (Input is an image of the game board, output is where to play the next stone)。

-   **图像分类器 (An Image Classifier)**
    -   不像对数字列表进行排序，没有明显的方法来硬编码识别猫或其他类别的算法 (Unlike sorting a list of numbers, there's no obvious way to hard-code the algorithm for recognizing a cat, or other classes)。
    -   **传统方法尝试 (You could try...)**
        -   寻找边缘 (Find edges)。
        -   寻找角点 (Find corners)。
        -   硬编码规则 (Hard-code rules)。
        -   这种方法很“脆弱”且不可扩展 (This approach is "brittle" and not scalable)。
    -   **机器学习：数据驱动方法 (Machine Learning: Data-Driven Approach)**
        1.  收集图像和标签的数据集 (Collect a dataset of images and labels)。
        2.  使用机器学习训练分类器 (Use Machine Learning to train a classifier)。
        3.  在新图像上评估分类器 (Evaluate the classifier on new images)。
        -   这种方法通过数据来“编程”计算机 (Program the computer via the data)。

-   **图像分类数据集 (Image Classification Datasets)**
    -   **MNIST**: 10个类别：数字0到9 (Digits 0 to 9)；28x28灰度图像 (28x28 grayscale images)；5万张训练图像 (50k training images), 1万张测试图像 (10k test images)。
    -   **CIFAR-10**: 10个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)；32x32 RGB图像 (32x32 RGB images)；5万张训练图像 (5k per class), 1万张测试图像 (1k per class)。
    -   **CIFAR-100**: 100个类别 (100 classes)；32x32 RGB图像 (32x32 RGB images)；5万张训练图像 (500 per class), 1万张测试图像 (100 per class)；有20个超类别，每个包含5个子类别 (20 superclasses with 5 classes each)。
    -   **ImageNet**: 1000个类别 (1000 classes)；约130万张训练图像 (~1.3M training images), 5万张验证图像 (50K validation images), 10万张测试图像 (100K test images)；性能指标 (Performance metric): Top-5 准确率 (Top-5 accuracy)。
    -   **MIT Places**: 365个不同场景类型 (365 classes of different scene types)；约800万张训练图像 (8M training images), 1.825万张验证图像 (50 per class), 32.85万张测试图像 (900 per class)。
    -   **Omniglot**: 1623个类别：来自50种不同字母表的字符 (1623 categories: characters from 50 different alphabets)；每个类别20张图像 (20 images per category)；旨在测试少样本学习 (Meant to test few-shot learning)。

-   **第一个分类器：最近邻 (First Classifier: Nearest Neighbor)**
    -   **训练函数 (train function)**
        -   训练速度 (Training speed): O(1) (常数时间)。
    -   **预测函数 (predict function)**
        -   预测速度 (Testing speed): O(N) (线性时间)。
        -   **问题 (Problem)**: 训练快但测试慢是糟糕的！我们需要快速测试！(This is bad: We can afford slow training, but we need fast testing!)。
        -   存在许多用于快速/近似最近邻的方法 (There are many methods for fast / approximate nearest neighbors)。

-   **最近邻分类器效果如何？(What does Nearest Neighbor look like?)**
    -   **L1 距离来比较图像 (L1 Distance to Compare Images)**
        -   计算对应像素的绝对值差之和 ($d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$ )。
    -   **视觉相似性与语义相似性 (Visual vs. Semantic Similarity)**
        -   最近邻往往是视觉上非常相似的图像 (Nearest neighbors tend to be very visually similar images)。
        -   但视觉相似性并不总是意味着语义相似性 (But visual similarity does not always mean semantic similarity)。
        -   例如：一个橙色斑点可能是青蛙，但其最近邻可能是一只猫 (e.g., an orange blob that is a frog, its nearest neighbor is a cat)。
    -   **最近邻决策边界 (Nearest Neighbor Decision Boundaries)**
        -   点是训练样本；颜色代表训练标签 (Points are training examples; colors give training labels)。
        -   背景颜色代表测试点将被分配的类别 (Background colors give the category a test point would be assigned)。
        -   决策边界 (Decision boundary): 两个分类区域之间的边界 (Boundary between two classification regions)。
        -   决策边界可能嘈杂，受离群点影响 (Decision boundaries can be noisy; affected by outliers)。
        -   如何平滑决策边界？使用更多的邻居！(How to smooth out decision boundaries? Use more neighbors!)。

-   **K-最近邻 (K-Nearest Neighbors)**
    -   不复制最近邻的标签，而是从 K 个最近点中取多数票 (Instead of copying label from nearest neighbor, take majority vote from K closest points)。
    -   K=1 (原始最近邻) vs. K=3 (更平滑的决策边界，受噪声影响小) (K=1 (original nearest neighbor) vs. K=3 (smoother boundaries, less affected by noise))。
    -   当 K > 1 时，类别之间可能存在平局，需要某种方法来打破平局 (When K > 1 there can be ties between classes. Need to break somehow!)。
    -   **距离度量 (Distance Metric)**
        -   L1 (曼哈顿) 距离 (L1 (Manhattan) distance): $d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$
        -   L2 (欧几里得) 距离 (L2 (Euclidean) distance): $d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$
        -   通过选择合适的距离度量，我们可以将 K-最近邻应用于任何类型的数据！(With the right choice of distance metric, we can apply K-Nearest Neighbor to any type of data!)。
        -   示例：使用tf-idf相似度比较研究论文 (Example: Compare research papers using tf-idf similarity)。

-   **超参数 (Hyperparameters)**
    -   什么是最佳的 K 值？(What is the best value of K to use?)。
    -   什么是最佳的距离度量？(What is the best distance metric to use?)。
    -   这些是超参数的例子：关于我们学习算法的选择，我们不从训练数据中学习；相反，我们在学习过程开始时设置它们 (These are examples of hyperparameters: choices about our learning algorithm that we don't learn from the training data; instead we set them at the start of the learning process)。
    -   它们非常依赖于问题 (Very problem-dependent)。
    -   通常需要尝试所有这些方法，看看哪种最适合我们的数据/任务 (In general need to try them all and see what works best for our data / task)。

-   **设置超参数 (Setting Hyperparameters)**
    -   **想法 #1**: 选择在数据上表现最好的超参数 (Idea #1: Choose hyperparameters that work best on the data)。
        -   **不好**: K=1 总是能在训练数据上完美运行 (BAD: K=1 always works perfectly on training data)。
        -   这会导致过拟合，对新数据没有泛化能力 (This leads to overfitting and no generalization to new data)。
    -   **想法 #2**: 将数据分割为训练集和测试集，选择在测试集上表现最好的超参数 (Idea #2: Split data into train and test, choose hyperparameters that work best on test data)。
        -   **不好**: 不知道算法在新数据上将如何表现 (BAD: No idea how algorithm will perform on new data)。
        -   因为我们已经使用测试集来选择超参数，测试集不再是“未见过”的数据 (Because we used the test set to select hyperparameters, it is no longer unseen data)。
        -   这是一个机器学习模型中“作弊”的常见错误 (This is a fundamental cardinal sin in machine learning models)。
    -   **想法 #3**: 将数据分割为训练集、验证集和测试集；在验证集上选择超参数，并在测试集上进行评估 (Idea #3: Split data into train, val, and test; choose hyperparameters on val and evaluate on test)。
        -   **更好！(Better!)**
        -   训练集 (train): 用于训练模型 (Used to train the model)。
        -   验证集 (validation): 用于选择超参数 (Used to select hyperparameters)。
        -   测试集 (test): 仅在所有决策完成后使用一次 (Used only once at the very end to evaluate the final model)。
    -   **想法 #4**: 交叉验证 (Cross-Validation)。将数据分割成多个折叠 (folds)，将每个折叠作为验证集进行尝试，并平均结果 (Split data into folds, try each fold as validation and average the results)。
        -   对小型数据集有用 (Useful for small datasets)。
        -   但在深度学习中不幸没有被频繁使用 (But unfortunately not used too frequently in deep learning)。
        -   **示例**: K 值的5折交叉验证 (Example of 5-fold cross-validation for the value of k)。
            -   每个点：单个结果 (Each point: single outcome)。
            -   线穿过平均值，条形表示标准差 (The line goes through the mean, bars indicated standard deviation)。
            -   (似乎 K~7 对此数据效果最好) (Seems that K~7 works best for this data)。

-   **K-最近邻：通用逼近 (K-Nearest Neighbor: Universal Approximation)**
    -   随着训练样本数量趋于无穷大，最近邻可以表示任何(*)函数！(As the number of training samples goes to infinity, nearest neighbor can represent any(*) function!)。
    -   (*) 须符合许多技术条件。仅在紧凑域上的连续函数；需要对训练点的间距等做出假设 (Subject to many technical conditions. Only continuous functions on a compact domain; need to make assumptions about spacing of training points; etc.)。
    -   **问题：维度诅咒 (Problem: Curse of Dimensionality)**
        -   为了均匀覆盖空间，所需的训练点数量随维度呈指数增长 (For uniform coverage of space, number of training points needed grows exponentially with dimension)。
        -   维度 = 1，点 = 4 (Dimensions = 1, Points = 4)。
        -   维度 = 2，点 = $4^2$ (Dimensions = 2, Points = $4^2$)。
        -   维度 = 3，点 = $4^3$ (Dimensions = 3, Points = $4^3$)。
        -   32x32 二值图像的可能数量约为 $2^{32 \times 32} \approx 10^{308}$ (Number of possible 32x32 binary images: $2^{32 \times 32} \approx 10^{308}$)。
        -   可见宇宙中基本粒子的数量约为 $10^{97}$ (Number of elementary particles in the visible universe: $\approx 10^{97}$)。
        -   这意味着我们永远无法收集足够的数据来密集覆盖整个图像空间 (This means we can never collect enough data to densely cover the entire space of images)。
    -   **K-最近邻在原始像素上很少使用 (K-Nearest Neighbor on raw pixels is seldom used)**
        -   在测试时非常慢 (Very slow at test time)。
        -   像素上的距离度量不具有信息性 (Distance metrics on pixels are not informative)。
            -   原始图像与修改后的图像在 L2 距离上可能相同，但语义上差异巨大 (Original image vs. boxed/shifted/tinted images have same L2 distance but are semantically different)。
        -   **最近邻与 ConvNet 特征结合效果良好！(Nearest Neighbor with ConvNet features works well!)**
            -   示例：图像标注与最近邻 (Example: Image Captioning with Nearest Neighbor)。
            -   通过深度卷积神经网络 (ConvNet) 提取的特征向量能够更好地捕捉图像的语义相似性 (Feature vectors computed from deep ConvNets can capture semantic similarity better)。

### 二、关键术语定义 (Key Term Definitions)

-   **图像分类 (Image Classification)**: 将给定图像分配到预定义的固定类别集合中的一个计算机视觉任务。
-   **语义鸿沟 (Semantic Gap)**: 指计算机处理的原始像素数据与人类对图像的语义理解之间的差异。
-   **视点变化 (Viewpoint Variation)**: 由于相机角度、位置等变化导致的同一物体在不同图像中像素值差异大的问题。
-   **类内差异 (Intraclass Variation)**: 同一类别内不同个体（例如不同品种的猫）在视觉外观上存在的巨大差异。
-   **细粒度类别 (Fine-Grained Categories)**: 视觉上非常相似但属于不同子类别的物体，如不同品种的猫或狗。
-   **背景杂乱 (Background Clutter)**: 图像中物体与背景融合，或背景元素干扰物体识别的情况。
-   **光照变化 (Illumination Changes)**: 场景光照条件改变导致图像像素值发生巨大变化，但物体本身语义不变。
-   **形变 (Deformation)**: 物体以不同姿态或形状出现在图像中，保持其类别但视觉表现多样。
-   **遮挡 (Occlusion)**: 物体部分被其他物体遮挡，导致其在图像中不完全可见。
-   **目标检测 (Object Detection)**: 识别图像中物体的位置（通常用边界框表示）并分类它们。
-   **图像标注 (Image Captioning)**: 根据图像内容生成一段描述性的自然语言文本。
-   **数据驱动方法 (Data-Driven Approach)**: 一种机器学习范式，通过从大量数据中学习模式来训练模型，而不是通过硬编码规则。
-   **MNIST**: 一个包含手写数字图像的经典图像分类数据集，常用于机器学习算法的初步测试。
-   **CIFAR-10**: 一个包含10个常见物体类别的彩色图像数据集，比MNIST更具挑战性。
-   **CIFAR-100**: CIFAR-10的扩展版本，包含100个类别。
-   **ImageNet**: 一个大规模图像数据库，包含数百万张图像和数千个类别，是图像分类任务的黄金标准基准。
-   **MIT Places**: 一个专注于场景识别的大型图像数据集。
-   **Omniglot**: 一个旨在测试少样本学习的数据集，包含来自多种语言的字符。
-   **Top-5 准确率 (Top-5 Accuracy)**: 一种评估指标，如果算法对图像预测的前5个标签中包含正确标签，则认为预测正确。
-   **少样本学习 (Few-Shot Learning)**: 机器学习的一个研究领域，旨在使算法能够从每个类别很少的训练样本中学习和泛化。
-   **最近邻 (Nearest Neighbor)**: 一种简单的分类算法，通过查找测试样本在训练集中最相似的样本的标签来进行预测。
-   **L1 距离 (L1 Distance)**: 也称为曼哈顿距离，用于比较两个图像的距离，计算对应像素绝对值差的总和。
-   **L2 距离 (L2 Distance)**: 也称为欧几里得距离，用于比较两个图像的距离，计算对应像素差的平方和的平方根。
-   **决策边界 (Decision Boundary)**: 在分类任务中，不同类别预测区域之间的界限。
-   **K-最近邻 (K-Nearest Neighbors)**: 最近邻算法的扩展，通过从 K 个最近点的多数票来预测标签，有助于平滑决策边界和减少离群点影响。
-   **超参数 (Hyperparameters)**: 在学习过程开始时设定的算法参数，不能直接从训练数据中学习。
-   **验证集 (Validation Set)**: 数据集中用于调整模型超参数的部分，独立于训练集和测试集。
-   **交叉验证 (Cross-Validation)**: 一种更稳健的超参数选择和模型评估技术，通过将数据分割成多个折叠并迭代使用不同折叠作为验证集来平均结果。
-   **维度诅咒 (Curse of Dimensionality)**: 在高维空间中，为了均匀覆盖空间，所需训练样本数量呈指数级增长的问题。
-   **tf-idf (term frequency-inverse document frequency)**: 一种用于文本分析的相似度度量，通过词频和逆文档频率来评估词语的重要性。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **机器学习系统通用框架 (General Framework for Machine Learning Systems)**:
    1.  收集图像和标签的数据集 (Collect a dataset of images and labels)。
    2.  使用机器学习训练分类器 (Use Machine Learning to train a classifier)。
    3.  在新图像上评估分类器 (Evaluate the classifier on new images)。

-   **训练函数签名 (Train Function Signature)**:
    ```python
    def train(images, labels):
        # 机器学习算法的核心逻辑
        # Machine learning!
        return model # 返回训练好的模型
    ```

-   **预测函数签名 (Predict Function Signature)**:
    ```python
    def predict(model, test_images):
        # 使用模型进行预测
        # Use model to predict labels
        return test_labels # 返回预测的标签
    ```

-   **L1 距离计算 (L1 Distance Calculation)**:
    -   $d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$
    -   将测试图像和训练图像的对应像素值相减，取绝对值，然后将所有结果相加。

-   **L2 距离计算 (L2 Distance Calculation)**:
    -   $d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$
    -   将测试图像和训练图像的对应像素值相减，取平方，然后将所有结果相加，最后开平方。

-   **最近邻分类器实现 (Nearest Neighbor Classifier Implementation)**:
    ```python
    import numpy as np

    class NearestNeighbor:
        def __init__(self):
            pass

        def train(self, X, y):
            """ X is N x D where each row is an example. Y is 1-dimension of size N
                *** the nearest neighbor classifier simply remembers all the training data ***
            """
            self.Xtr = X
            self.ytr = y

        def predict(self, X):
            """ X is N x D where each row is an example we wish to predict label for """
            num_test = X.shape[0]
            # 确保输出类型与输入类型匹配
            # lets make sure that the output type matches the input type
            Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

            # 遍历所有测试行
            # loop over all test rows
            for i in xrange(num_test):
                # 使用L1距离(绝对值差之和)找到第i个测试图像的最近训练图像
                # find the nearest training image to the i'th test image
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
                min_index = np.argmin(distances) # 获取距离最小的索引
                Ypred[i] = self.ytr[min_index] # 预测最近邻样本的标签

            return Ypred
    ```

### 四、讲师提出的思考题 (Questions Posed by the Instructor)

-   有 N 个样本时，训练速度有多快？(With N examples, how fast is training?)
-   有 N 个样本时，测试速度有多快？(With N examples, how fast is testing?)
-   如何平滑决策边界？(How to smooth out decision boundaries?)
-   什么是最佳的 K 值？(What is the best value of K to use?)
-   什么是最佳的距离度量？(What is the best distance metric to use?)
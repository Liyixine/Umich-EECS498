### [📚] 视频学习脚手架: Lecture 15 | Object Detection

### 一、核心内容大纲 (Core Content Outline)
-   **讲座引言 (Lecture Introduction)**
    -   本讲主题：目标检测 (Object Detection)
    -   上次回顾 (Assumed from previous segment): 可视化与理解卷积神经网络 (Visualizing and Understanding CNNs)
-   **计算机视觉任务概述 (Overview of Computer Vision Tasks)**
    -   图像分类 (Image Classification)
    -   语义分割 (Semantic Segmentation)
    -   目标检测 (Object Detection)
        -   任务定义 (Task Definition): 输入单张 RGB 图像，输出一组检测到的对象，每个对象包含类别标签和边界框。
        -   挑战 (Challenges):
            -   多重输出 (Multiple outputs)：可变数量的对象，每个对象有多个属性（类别、位置）。
            -   输出类型多样 (Multiple types of output)：同时预测“是什么”(类别) 和“在哪里”(位置)。
            -   大图像 (Large images)：需要处理更高分辨率的图像，计算量大。
    -   实例分割 (Instance Segmentation)
-   **检测单个对象 (Detecting a Single Object)**
    -   架构 (Architecture): 输入图像 -> 卷积网络 (ConvNet) -> 特征向量 (Feature Vector)
        -   “是什么”分支 (“What” branch): 全连接层 (Fully Connected) -> 类别得分 (Class Scores) -> Softmax Loss。
        -   “在哪里”分支 (“Where” branch): 全连接层 (Fully Connected) -> 边界框坐标 (Box Coordinates (x, y, w, h)) -> L2 Loss (将定位视为回归问题)。
    -   多任务损失 (Multitask Loss): Softmax Loss 和 L2 Loss 的加权和。
    -   迁移学习 (Transfer Learning): ConvNet 通常在 ImageNet 上预训练。
    -   问题 (Problem): 图像中可能存在多个对象。
-   **检测多个对象：早期方法 (Detecting Multiple Objects: Early Approaches)**
    -   滑动窗口 (Sliding Window): 对图像中的许多不同裁剪区域 (crops) 应用 CNN。
        -   问题 (Problem): 可能的边界框数量巨大，计算量难以承受。
    -   区域提议 (Region Proposals): 寻找一小部分可能覆盖所有对象的边界框（例如使用启发式算法如选择性搜索，生成约 2000 个提议）。
-   **R-CNN: 基于区域的 CNN (Region-Based CNN)**
    -   测试时流程 (Test-time pipeline):
        1.  运行区域提议方法 (Region Proposal Method) 生成约 2000 个感兴趣区域 (RoI)。
        2.  将每个 RoI 调整大小 (resize) 到固定尺寸（例如 224x224）。
        3.  每个调整后的区域独立通过卷积网络 (ConvNet) 运行。
        4.  ConvNet 预测类别得分和边界框变换 (Bounding Box Regression)。
        5.  使用得分选择区域提议的子集作为输出。
-   **比较边界框：交并比 (Comparing Boxes: Intersection over Union (IoU))**
    -   定义和公式 (Definition and Formula): IoU = $\frac{\text{Area of Intersection}}{\text{Area of Union}}$
    -   又称 Jaccard 相似度 (Jaccard similarity) 或 Jaccard 指数 (Jaccard index)。
    -   性质 (Properties): 结果始终在 0 和 1 之间。
        -   1: 完美重合。
        -   0: 完全不重叠。
    -   IoU 值解释 (Interpretation of IoU values):
        -   IoU > 0.5: 通常被认为是“可以接受 (decent)”的检测。
        -   IoU > 0.7: 通常被认为是“相当好 (pretty good)”的检测。
        -   IoU > 0.9: 通常被认为是“几乎完美 (almost perfect)”的检测。
-   **重叠边界框：非极大值抑制 (Overlapping Boxes: Non-Max Suppression (NMS))**
    -   问题 (Problem): 目标检测器通常会输出许多重叠的检测框。
    -   解决方案 (Solution): 使用非极大值抑制 (NMS) 对原始检测结果进行后处理。
    -   NMS 算法步骤 (NMS Algorithm Steps):
        1.  选择当前所有边界框中得分最高的框。
        2.  消除所有与该框 IoU 大于某个阈值（例如 0.7）的得分较低的边界框。
        3.  如果仍有框剩余，返回步骤 1。
    -   NMS 的局限性 (Problem with NMS): NMS 在对象高度重叠（例如人群图像）的情况下可能会消除“好”的检测框，这是一个计算机视觉领域的开放性问题。
-   **评估目标检测器：平均精度均值 (Evaluating Object Detectors: Mean Average Precision (mAP))**
    -   mAP 是衡量目标检测器整体性能的标准指标。
    -   平均精度 (Average Precision (AP)) 计算（针对每个类别单独计算）:
        1.  在所有测试图像上运行目标检测器（应用 NMS）。
        2.  将所有检测结果按置信度得分从高到低排序。
        3.  对每个检测结果：
            *   如果它与某个未被匹配的真实边界框的 IoU 大于预设阈值（通常为 0.5），则标记为真阳性 (True Positive, TP) 并将该真实框从集合中移除。
            *   否则，标记为假阳性 (False Positive, FP)。
        4.  根据累积的 TP 和 FP 数量，在不同的置信度阈值下计算精确率 (Precision) 和召回率 (Recall)，并绘制精确率-召回率 (Precision-Recall, PR) 曲线。
        5.  AP 等于 PR 曲线下的面积。AP = 1.0 表示模型完美地检测到了所有目标且没有假阳性。
    -   平均精度均值 (mAP): 对所有类别的 AP 值进行平均。
    -   多阈值 mAP (mAP@threshold): 在不同的 IoU 阈值（如 mAP@0.5, mAP@0.75）下计算 mAP，然后取其平均值（例如 COCO 数据集的 mAP 为多个 IoU 阈值的平均值）。
-   **R-CNN 演进：解决速度问题 (R-CNN Evolution: Addressing Speed)**
    -   **"慢速" R-CNN ("Slow" R-CNN)**:
        -   问题 (Problem): 非常慢，每张图像需进行约 2000 次独立 CNN 前向传播，无法实时运行。
    -   **Fast R-CNN**:
        -   解决方案 (Solution): 在 warping 之前运行 CNN。
        1.  **主干网络 (Backbone network)**: 对整个输入图像运行一次全卷积的 CNN（例如 AlexNet, VGG, ResNet），获取图像特征 (Image features)。
        2.  从提议方法获取感兴趣区域 (RoIs)。
        3.  **特征裁剪与调整大小 (Crop + Resize features)**: 将 RoI 投影到图像特征图上，并对这些特征图区域进行裁剪和调整大小。
        4.  **每区域网络 (Per-Region Network)**: 将裁剪并调整大小后的特征输入到轻量级 CNN（通常是几个全连接层）中，以预测类别得分和边界框变换。
        -   优点 (Benefit): 大部分计算（主干网络）在所有提议之间共享，显著提高了速度。
        -   特征裁剪方法：
            -   **RoI Pooling (Region of Interest Pooling)**: 将提议框投影到特征图上，吸附 (snap) 到最近的网格单元，然后将每个区域划分为固定大小的网格，并在每个子区域内进行最大池化 (max-pooling)。
                -   问题 (Problem): 由于“吸附”操作，可能导致轻微不对齐 (slight misalignment)，且子区域大小可能不完全相等。
            -   **RoI Align (Region of Interest Align)**: 改进版，不进行“吸附”，而是通过双线性插值 (bilinear interpolation) 在每个子区域的规则间隔点采样特征，保证更好的对齐和可微分性。
    -   **"更快" R-CNN (Faster R-CNN)**: 可学习的区域提议 (Learnable Region Proposals)。
        -   问题 (Problem): Fast R-CNN 的运行时瓶颈转移到区域提议的计算（通常仍在 CPU 上运行）。
        -   解决方案 (Solution): 引入**区域提议网络 (Region Proposal Network, RPN)**，从主干网络的特征图直接预测区域提议。RPN 与主干网络共享计算。
        -   RPN 自身也有两个损失：分类损失和回归损失。
            *   RPN 分类 (RPN classification): 预测每个**锚框 (anchor box)** 是否包含对象 (object / not object)。锚框是预定义在特征图上不同位置、大小和长宽比的固定大小边界框。
            *   RPN 回归 (RPN regression): 预测从锚框到更精确提议框的变换。
        -   "更快" R-CNN 的训练涉及 4 个损失联合训练：RPN 分类、RPN 回归、最终对象分类、最终对象回归。
        -   **性能 (Performance)**: "更快" R-CNN 比 Fast R-CNN 快得多，因为它解决了区域提议的 CPU 瓶颈，实现了接近实时的目标检测。
-   **单阶段目标检测 (Single-Stage Object Detection)**
    -   Fast R-CNN 和 Faster R-CNN 是两阶段 (Two-stage) 检测器：第一阶段生成提议，第二阶段对提议进行分类和回归。
    -   **单阶段检测器 (Single-Stage Detector)**：取消了区域提议阶段，直接在特征图上同时预测所有对象的类别和边界框。
    -   RPN 分类通常被扩展，直接将每个锚框分类为 C 个对象类别中的一个（或背景）。
    -   通常会预测类别特定的边界框回归 (category-specific regression)。
    -   例子：YOLO (You Only Look Once), SSD (Single Shot Detector), RetinaNet。
-   **目标检测性能与挑战 (Object Detection Performance and Challenges)**
    -   **变量繁多 (Lots of variables!)**:
        -   主干网络 (Backbone network) 选择 (e.g., Inception, MobileNet, ResNet, VGG)。
        -   元架构 (Meta Architecture) 选择 (e.g., Faster R-CNN, R-FCN, SSD)。
        -   图像分辨率，裁剪分辨率，锚框数量、尺寸和长宽比，IoU 阈值等大量超参数。
    -   **要点 (Takeaways)**:
        -   两阶段方法 (Faster R-CNN) 通常能获得最佳准确率 (accuracy)，但速度较慢 (slower)。
        -   单阶段方法 (SSD) 速度更快，但性能可能稍逊 (don't perform as well)。
        -   更大的主干网络 (Bigger backbones) 通常能提高性能，但速度会变慢。
        -   **近年性能提升技巧 (Improved performance with many tricks in recent years)**:
            -   训练时间更长 (Train longer!)。
            -   多尺度主干 (Multiscale backbone): 例如特征金字塔网络 (Feature Pyramid Networks)。
            -   更好的主干 (Better backbone): 例如 ResNeXt。
            -   单阶段方法性能显著提高 (Single-stage methods have improved)！
            -   使用非常大的模型 (Very big models work better)。
            -   测试时数据增强 (Test-time augmentation) 和集成 (ensembles) 可进一步提高性能。
    -   **当前SOTA (Current State-of-the-Art)**: 已经达到非常高的 mAP，甚至超过了 55 mAP（在 COCO 数据集上）。
    -   **建议 (General Advice)**: 目标检测很难，不要自己实现 (Don't implement it yourself!)，除非你是为了学习或作业。
    -   **开源代码 (Open-Source Code)**:
        -   TensorFlow Detection API (Google): 包含 Faster R-CNN, SSD, RFCN, Mask R-CNN 等。
        -   Detectron2 (PyTorch) (Facebook): 包含 Fast/Faster/Mask R-CNN, RetinaNet 等。

### 二、关键术语定义 (Key Term Definitions)
-   **目标检测 (Object Detection)**: 一种计算机视觉任务，旨在识别图像中的多个对象实例，并提供每个对象的类别标签和精确的空间位置（通常是边界框）。
-   **边界框回归 (Bounding Box Regression)**: 预测从一个初始边界框（例如区域提议或锚框）到更精确的目标对象真实边界框的微调变换。
-   **交并比 (Intersection over Union, IoU)**: 一种衡量两个边界框重叠程度的指标，计算方式为两个框的交集面积除以它们的并集面积。
-   **非极大值抑制 (Non-Max Suppression, NMS)**: 一种后处理算法，用于在目标检测中消除重复的、高度重叠的检测框，只保留每个对象得分最高的检测框。
-   **平均精度均值 (Mean Average Precision, mAP)**: 目标检测领域常用的评估指标，是所有类别平均精度 (AP) 的平均值，综合衡量了检测的准确性和召回率。
-   **精确率-召回率曲线 (Precision-Recall (PR) Curve)**: 在不同分类阈值下，以召回率为横轴，精确率为纵轴绘制的曲线，用于评估分类或检测模型的性能。
-   **区域提议 (Region Proposals)**: 从图像中生成一小组可能包含对象的候选边界框，以减少后续目标检测的计算量。
-   **R-CNN (Region-Based CNN)**: 一种早期基于区域提议的深度学习目标检测框架。
-   **Fast R-CNN**: R-CNN 的改进版，通过共享卷积特征图，显著提高了检测速度。
-   **RoI Pooling (Region of Interest Pooling)**: Fast R-CNN 中的一个操作，用于将不同大小的 RoI 区域映射到固定大小的特征向量，以便输入到全连接层。
-   **RoI Align (Region of Interest Align)**: RoI Pooling 的改进版，通过精确计算 RoI 在特征图上的位置并使用双线性插值，避免了量化误差，提高了准确性。
-   **Faster R-CNN**: Fast R-CNN 的改进版，引入了**区域提议网络 (Region Proposal Network, RPN)** 来学习生成区域提议，从而进一步提高了检测速度和准确性。
-   **区域提议网络 (Region Proposal Network, RPN)**: Faster R-CNN 中的一个子网络，它是一个全卷积网络，用于直接从特征图预测区域提议，并判断提议是否包含对象。
-   **锚框 (Anchor Box)**: 在 RPN 中使用的预定义边界框集合，它们具有不同的尺寸和长宽比，被放置在特征图的每个位置，用于预测可能包含对象的区域。
-   **单阶段目标检测 (Single-Stage Object Detection)**: 不区分区域提议和最终分类/回归阶段，直接在单个网络中完成所有检测任务的方法（例如 YOLO, SSD, RetinaNet）。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **边界框坐标变换 (Bounding Box Coordinate Transformation) 公式**:
    假设区域提议框为 $(p_x, p_y, p_w, p_h)$，模型预测的变换参数为 $(t_x, t_y, t_w, t_h)$，则输出的最终边界框 $(b_x, b_y, b_w, b_h)$ 计算如下：
    -   平移（相对于框大小）：
        $b_x = p_x + p_w t_x$
        $b_y = p_y + p_h t_y$
    -   对数空间尺度变换：
        $b_w = p_w \exp(t_w)$
        $b_h = p_h \exp(t_h)$

-   **非极大值抑制 (Non-Max Suppression (NMS)) 算法**:
    1.  选择当前所有边界框中得分最高的框。
    2.  计算该最高得分框与其他所有框的交并比 (IoU)。
    3.  消除所有与最高得分框 IoU 大于某个阈值（例如 0.7）且得分低于最高得分框的框。
    4.  如果仍有框剩余，则回到步骤 1，直到所有框都被处理。

-   **平均精度 (Average Precision, AP) 计算步骤**:
    1.  在所有测试图像上运行目标检测器（应用 NMS）。
    2.  对所有检测结果，按置信度得分从高到低排序。
    3.  对于每个检测结果（按排序顺序）：
        *   如果它与某个未被匹配的真实边界框的 IoU 大于预设阈值（例如 0.5），则标记为真阳性 (True Positive, TP) 并将该真实框从集合中移除（确保每个真实框只匹配一次）。
        *   否则，标记为假阳性 (False Positive, FP)。
        *   计算当前考虑的所有检测框的累积精确率和召回率。
    4.  根据计算出的精确率和召回率点，绘制 PR 曲线。
    5.  AP 等于 PR 曲线下的面积。

-   **Faster R-CNN 损失函数**:
    Faster R-CNN 联合训练 (Jointly train) 包含 4 个损失：
    1.  **RPN 分类损失 (RPN classification)**: 锚框是否是对象 / 不是对象。
    2.  **RPN 回归损失 (RPN regression)**: 预测从锚框到提议框的变换。
    3.  **对象分类损失 (Object classification)**: 将提议分类为背景 / 对象类别。
    4.  **对象回归损失 (Object regression)**: 预测从提议框到对象框的变换。

-   **开源目标检测代码库 (Open-Source Object Detection Codebases)**:
    -   **TensorFlow Detection API (Google)**:
        -   GitHub 链接: `https://github.com/tensorflow/models/tree/master/research/object_detection`
        -   包含的模型：Faster R-CNN, SSD, RFCN, Mask R-CNN 等。
    -   **Detectron2 (PyTorch) (Facebook)**:
        -   GitHub 链接: `https://github.com/facebookresearch/detectron2`
        -   包含的模型：Fast/Faster/Mask R-CNN, RetinaNet 等。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   如果不对重叠的检测框进行处理，你最终会得到什么结果？(What if you just could do this and you had like infinite compute and you just ran all 58 million of them, what do you think would happen?)
-   我们真的需要第二阶段吗？(Do we really need the second stage?) (指 Fast R-CNN 的两阶段架构，在 Faster R-CNN 之后引出单阶段检测器)

---

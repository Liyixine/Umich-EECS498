### [📚] 视频学习脚手架: Lecture 17: 3D Vision

### 一、核心内容大纲 (Core Content Outline)
-   **引言与2D视觉回顾 (Introduction & 2D Vision Recap)**
    -   本讲主题：3D视觉 (3D Vision) [00:00]
    -   回顾2D形状预测任务 (Predicting 2D Shapes of Objects) [00:07]
        -   分类 (Classification): 识别图像中的主要对象。
        -   语义分割 (Semantic Segmentation): 为图像中的每个像素分配类别标签（如草地、猫、树、天空）。
        -   目标检测 (Object Detection): 识别图像中的多个对象并用边界框定位（如狗、狗、猫）。
        -   实例分割 (Instance Segmentation): 识别图像中的每个对象实例，并给出精确的像素级掩码（如狗、狗、猫）。
        -   其他任务：关键点估计 (Keypoint Estimation)。
    -   2D任务特点：主要关注图像内的识别和定位，不涉及空间深度信息。
    -   向3D视觉的过渡 (Transition to 3D Vision) [01:02]
        -   现实世界是三维的 (The world we live in is three-dimensional)。
        -   目标：使神经网络模型能够处理和理解三维空间信息 (add this third spatial dimension to our neural network models)。

-   **3D视觉的两个核心问题 (Two Core Problems in 3D Vision)** [01:31]
    -   **问题一：从单张图像预测3D形状 (Predicting 3D Shapes from single image)**
        -   输入：单张RGB图像 (single RGB image)。
        -   输出：对象的三维形状表示 (representation of the 3D shape of objects)。
    -   **问题二：处理3D输入数据 (Processing 3D input data)**
        -   输入：三维数据 (3D data)。
        -   输出：基于三维输入的分类或分割决策 (classification decision or some segmentation decision)。
    -   假设：今日讨论均基于全监督任务 (fully supervised task)，即有完整的训练集包含输入和对应的真实标签。

-   **3D视觉的更多主题 (Many More Topics in 3D Vision)** [03:02]
    -   计算对应关系 (Computing correspondences)
    -   多视角立体 (Multi-view stereo)
    -   运动结构 (Structure from Motion)
    -   同步定位与地图构建 (Simultaneous Localization and Mapping, SLAM)
    -   自监督学习 (Self-supervised learning)
    -   视图合成 (View Synthesis)
    -   可微分图形 (Differentiable graphics)
    -   3D传感器 (3D Sensors)
    -   注：3D视觉领域有许多非深度学习方法仍然活跃且重要 (Many non-Deep Learning methods alive and well in 3D!)，因为涉及到大量的几何学知识。

-   **3D形状表示 (3D Shape Representations)** [04:57, 31:02]
    -   **深度图 (Depth Map)** [05:37]
        -   定义：为每个像素提供从相机到该像素处世界中对象的距离 (distance from the camera to the object in the world at that pixel)。
        -   RGB图像 + 深度图像 = RGB-D图像 (2.5D) (RGB image + Depth image = RGB-D Image (2.5D))。
        -   获取方式：可直接通过某些3D传感器（如Microsoft Kinect）记录。
        -   局限性：无法捕捉被遮挡对象 (cannot capture occluded objects)。
        -   **预测深度图 (Predicting Depth Maps)** [08:09]
            -   架构：全卷积网络 (Fully Convolutional network)。
            -   损失函数：逐像素L2距离损失 (Per-pixel Loss (L2 Distance))。
            -   问题：尺度/深度模糊性 (Scale / Depth Ambiguity) [09:47]
                -   定义：小而近的物体与大而远的物体在单张图像中看起来完全相同 (A small, close object looks exactly the same as a larger, farther-away object)。绝对尺度/深度是模糊的 (Absolute scale / depth are ambiguous from a single image)。
            -   解决方案：尺度不变损失 (Scale invariant loss) [10:54]
                -   损失函数设计使其不惩罚全局尺度上的偏差 (does not penalize a global multiplicative offset in scale)。
    -   **表面法线 (Surface Normals)** [12:10]
        -   定义：为每个像素提供一个向量，表示该像素处世界中对象的法向量 (a vector giving the normal vector to the object in the world for that pixel)。
        -   表示：通常使用RGB颜色来绘制法线图。
        -   **预测法线 (Predicting Normals)** [13:35]
            -   架构：与预测深度图类似的全卷积网络。
            -   损失函数：逐像素点积损失 (Per-pixel Loss: $(x \cdot y) / (|x| |y|)$) (余弦相似度)。
            -   优势：比深度图提供更多几何信息。
            -   局限性：与深度图类似，无法表示被遮挡部分。
            -   可训练联合网络 (joint network) 同时进行语义分割、深度估计和法线估计。
    -   **体素网格 (Voxel Grid)** [15:05, 31:02]
        -   定义：用V x V x V的占用率网格来表示形状 (Represent a shape with a V x V x V grid of occupancies)。类似于3D的分割掩码 (segmentation masks in Mask R-CNN, but in 3D!)。
        -   优势：概念上简单 (Conceptually simple: just a 3D grid!)。
        -   劣势：需要高空间分辨率来捕捉精细结构 (Need high spatial resolution to capture fine structures)，扩展到高分辨率非同小可 (Scaling to high resolutions is nontrivial!) (内存占用量大，1024^3体素网格占用4GB内存) [27:58]。
        -   **处理体素输入：3D卷积 (Processing Voxel Inputs: 3D Convolution)** [16:55]
            -   架构：3D ShapeNets (基于论文 "3D ShapeNets: A Deep Representation for Volumetric Shapes", CVPR 2015)。
            -   输入：体素网格 (1 x 30 x 30 x 30)，表示空间中每个点的占用状态 (binary occupancy)。
            -   网络结构：多层3D卷积层 (如6x6x6, 5x5x5, 4x4x4 卷积核)，特征维度逐渐增加，空间维度逐渐减小。
            -   末端：全连接层 (FC Layer) 预测类别分数 (Class Scores)。
            -   训练：分类损失函数 (classification loss)。
        -   **生成体素形状：3D卷积 (Generating Voxel Shapes: 3D Convolution)** [21:05]
            -   任务：输入2D图像，输出3D体素网格。
            -   架构：2D CNN (处理输入图像) -> 全连接层 (FC Layer, 将2D特征转换为3D特征) -> 3D CNN (带有空间上采样，增加体素分辨率) -> 体素输出。
            -   训练：逐体素交叉熵损失 (per-voxel cross-entropy loss)。
            -   计算成本高：3D卷积操作的计算量与空间尺寸的立方成正比。
        -   **体素扩展方法 (Scaling Voxels)**:
            -   **八叉树 (Oct-Trees)** [29:15]
                -   使用具有异构分辨率的体素网格 (Use voxel grids with heterogeneous resolution!)。
                -   在低分辨率下捕捉粗略结构，在需要细节的地方使用更高分辨率。
            -   **嵌套形状层 (Nested Shape Layers)** [30:14]
                -   将形状预测为正空间和负空间的组合 (Predict shape as a composition of positive and negative spaces)。
                -   类似于俄罗斯套娃 (Matryoshka Russian dolls)。
                -   通过稀疏体素层从内向外表示形状 (Represent shape from inside out using sparse voxel layers)。
            -   **体素管 (Voxel Tubes)** [24:16]
                -   一种特殊的架构，只使用2D卷积进行体素生成，但牺牲了Z轴的平移不变性 (loses translational invariance in the Z dimension)。
                -   在2D CNN的最后一层，输出的V个滤波器被解释为沿深度维度的“体素管”分数。
    -   **隐式曲面 (Implicit Surface)** [31:11, 31:19]
        -   定义：学习一个函数 $o: \mathbb{R}^3 \rightarrow \{0, 1\}$，用于分类三维空间中的任意点是位于形状内部还是外部 (Learn a function to classify arbitrary 3D points as inside / outside the shape)。
        -   三维对象的表面是该函数的水平集 (The surface of the 3D object is the level set): $\{x : o(x) = 1/2\}$。
        -   别名：符号距离函数 (signed distance function, SDF)。SDF给出到形状表面的欧几里得距离 (Euclidean distance)，符号表示内部或外部。
        -   **实现与训练 (Implementation and Training)** [33:43]
            -   训练方式：神经网络将3D坐标作为输入，输出该点是位于形状内部还是外部的概率。
            -   多尺度输出 (multiscale outputs)：允许在不同分辨率下评估函数，类似于Oct-Trees。
            -   提取显式形状输出 (Extracting explicit shape outputs)：需要后处理。
                -   通常通过Marching Cubes算法在隐式函数上提取等值面来生成网格。
                -   生成的网格可以进一步简化 (simplify mesh) 和使用梯度细化 (refine using gradients)。
    -   **点云 (Point Cloud)** [31:02, 35:17]
        -   定义：用P个点的集合在三维空间中表示形状 (Represent shape as a set of P points in 3D space)。这些点通常覆盖对象表面。
        -   优势：可以表示精细结构而无需大量点 (Can represent fine structures without huge numbers of points)；更具适应性 (more adaptive)。
        -   劣势：需要新的架构和损失函数 (Requires new architecture, losses, etc)；不直接表示表面 (Doesn't explicitly represent the surface of the shape)；提取网格需要后处理 (extracting a mesh for rendering or other applications requires post-processing)。
        -   应用：自动驾驶汽车 (self-driving car applications) 中的LiDAR传感器数据常以点云形式表示。
        -   **处理点云输入：PointNet (Processing Pointcloud Inputs: PointNet)** [38:44]
            -   目标：将点云作为集合处理 (Want to process pointclouds as sets)，点的顺序不应影响结果 (order should not matter)。
            -   架构：输入点云 (P x 3) -> 对每个点运行MLP -> Max-Pool (最大池化) -> 全连接层 (Fully Connected)。
            -   MLP在每个点上独立运行，生成点特征 (P x D)。
            -   最大池化操作在所有点特征上进行，生成一个全局特征向量 (Pooled vector: D)，确保对点顺序的不变性。
            -   最后通过全连接层输出分类分数 (Class score: C)。
        -   **生成点云输出 (Generating Pointcloud Outputs)** [42:16]
            -   任务：输入单张RGB图像，输出点云。
            -   架构：2D CNN -> 图像特征 (Image Features) -> 两个分支：
                -   全连接分支 (Fully connected branch) -> P1个点 (P1 x 3)。
                -   卷积分支 (Convolutional branch) -> P2个点 (P2 x 3 x H' x W')。
                -   将两部分点合并得到最终点云 $((P1 + H'W'P2) \times 3)$。
        -   **预测点云：损失函数 (Predicting Point Clouds: Loss Function)** [45:05]
            -   需要一个可微分的方式来比较点云作为集合 (We need a (differentiable) way to compare pointclouds as sets!)。
            -   **Chamfer 距离 (Chamfer distance)**:
                -   定义：点云 $S_1$ 和 $S_2$ 之间的Chamfer距离是 $S_1$ 中每个点到 $S_2$ 中最近点的L2距离之和，加上 $S_2$ 中每个点到 $S_1$ 中最近点的L2距离之和。
                -   公式: $d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2^2 + \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2^2$。
                -   只有当两个点云完美重合时，距离才为0。
                -   问题：对异常值 (outliers) 非常敏感 (very sensitive to outliers!)。
            -   **F1 分数 (F1 Score)** [57:55]
                -   类似于Chamfer距离，从预测和真实形状表面采样点。
                -   精确度@t (Precision@t) = 预测点中在真实点某个阈值t内的比例。
                -   召回率@t (Recall@t) = 真实点中在预测点某个阈值t内的比例。
                -   F1@t = $2 \times (\text{Precision@t} \times \text{Recall@t}) / (\text{Precision@t} + \text{Recall@t})$。
                -   对异常值更鲁棒 (F1 score is robust to outliers!)。
                -   结论：F1分数可能是目前最常用的3D形状预测度量标准。
    -   **网格 (Mesh)** [31:02, 49:54]
        -   定义：将三维形状表示为三角形的集合 (Represent a 3D shape as a set of triangles)。包含顶点 (Vertices) 和面 (Faces)。
        -   优势：图形学中的标准表示 (Standard representation for graphics)；明确表示3D形状 (Explicitly represents 3D shapes)；具有适应性 (Adaptive) (可以高效表示平坦表面，为精细细节区域分配更多面)。
        -   可以在顶点上附加数据并通过整个表面进行插值 (Can attach data on verts and interpolate over the whole surface)：RGB颜色、纹理坐标、法线向量等。
        -   劣势：用神经网络处理非平凡 (Nontrivial to process with neural nets!)。
        -   **预测网格：Pixel2Mesh (Predicting Meshes: Pixel2Mesh)** [56:21]
            -   输入：单张RGB图像。输出：对象的三角形网格。
            -   **核心思想 (Key Ideas)**:
                1.  **迭代细化 (Iterative Refinement)** [57:33]: 从初始椭球体网格开始 -> 网络预测每个顶点的偏移量 -> 重复进行网格变形、图解池化等操作，逐步增加顶点数量并细化网格，使其与输入图像的几何形状匹配。
                2.  **图卷积 (Graph Convolution)** [58:59]:
                    -   输入：带有每个顶点特征向量的图。
                    -   新的顶点特征依赖于其自身特征和相邻顶点特征的加权和。
                    -   使用相同的权重计算所有输出，实现类似卷积的局部特征聚合。
                    -   能够处理具有任意拓扑结构的图。
                3.  **顶点对齐特征 (Vertex-Aligned Features)** [01:00:10]:
                    -   对于网格中的每个顶点，使用相机信息投影到图像平面。
                    -   使用双线性插值 (bilinear interpolation) 从2D CNN提取的图像特征图中采样对应位置的特征。
                    -   这类似于RoI-Align操作，保持输入图像和特征向量之间的对齐，将图像信息整合到3D网格中。
                4.  **Chamfer 损失函数 (Chamfer Loss Function)** [01:00:10]: 用于比较预测网格和真实网格。
                    -   通过对网格表面采样点来将其转换为点云，然后计算点云之间的Chamfer距离。
                    -   问题：需要在线采样 (Need to sample online!) (效率是关键)；需要通过采样进行反向传播 (Need to backprop through sampling!)。
        -   **Mesh R-CNN: 形状正则化 (Shape Regularizers)** [01:10:09]
            -   仅使用Chamfer损失会导致网格退化 (Using Chamfer as only mesh loss gives degenerate meshes)（例如，产生“破碎”的表面）。
            -   需要“网格正则化器” (mesh regularizer) 来鼓励良好的预测 (to encourage nice predictions!)：例如，最小化预测网格中边的L2范数 ($L_{edge}$ = minimize L2 norm of edges in the predicted mesh)。这有助于生成更平滑、结构更合理的网格。
        -   **Mesh R-CNN: 混合3D形状表示 (Hybrid 3D shape representation)** [01:13:58]
            -   结合体素预测和网格变形的优点。
            -   通过体素预测创建初始网格预测 (Use voxel predictions to create initial mesh prediction!)，解决网格变形方法无法改变拓扑结构（如孔洞）的限制。
            -   **Mesh R-CNN 流程 (Pipeline)** [01:15:35]:
                1.  2D物体识别 (2D object recognition): 输入图像 -> Mask R-CNN -> 输出边界框和掩码预测。
                2.  3D物体体素 (3D object voxels): 从2D预测中提取粗略的3D体素表示。
                3.  3D物体网格 (3D object meshes): 将体素转换为块状网格，然后进行迭代细化。
            -   结果：能够预测具有复杂拓扑结构（如书架上的孔洞）的网格。
            -   Amodal 补全 (Amodal completion): 预测对象的被遮挡部分 (predict occluded parts of objects)。
            -   局限性：分割失败 (segmentation failures) 会传播到网格 (propagate to meshes)。

-   **相机与坐标系 (Cameras & Coordinate Systems)** [01:02:59]
    -   相机系统在3D中变得复杂 (Cameras get complicated in 3D)。
    -   **规范坐标 (Canonical Coordinates)** [01:03:00]
        -   定义：在一个规范坐标系中预测3D形状 (Predict 3D shape in a canonical coordinate system)，例如，椅子的正面是+z方向，这与输入图像的视角无关。
        -   许多论文都在规范坐标中进行预测，因为数据加载更简单 (Many papers predict in canonical coordinates – easier to load data)。
        -   问题：规范视图打破了“特征对齐原则”(Canonical view breaks the "principle of feature alignment")，即预测应该与输入对齐。
    -   **视图坐标 (View Coordinates)** [01:05:46]
        -   定义：预测与相机视角对齐的3D形状 (Predict 3D shape aligned to the viewpoint of the camera)。
        -   视图坐标保持了输入和预测之间的对齐 (View coordinates maintain alignment between inputs and predictions!)。
        -   研究表明，在视图坐标系中训练的网络在测试时对已知形状的新视图、新模型和新类别具有更好的泛化能力。
        -   视图中心体素预测 (View-centric voxel predictions): 体素考虑透视相机，因此“体素”实际上是视锥 (frustums)。

-   **3D数据集 (3D Datasets)** [01:06:50]
    -   **以对象为中心的数据集 (Object-Centric Datasets)**:
        -   **ShapeNet**: 包含约50个类别和5万个3D CAD模型。标准划分有13个类别，约4.4万个模型，每个模型有25张渲染图像。
            -   缺点：合成的 (Synthetic)，孤立的对象 (isolated objects)；无上下文 (no context)。大量椅子、汽车、飞机（类别不平衡）。
        -   **Pix3D**: 9个类别，219个宜家家具3D模型，对齐到约1.7万张真实图像。
            -   优点：真实图像 (Real images!)；有上下文 (Context!)。
            -   缺点：小而局部 (Small, partial annotations)（每张图像只有一个对象注释）。

### 二、关键术语定义 (Key Term Definitions)
-   **3D视觉 (3D Vision)**: 计算机视觉领域的一个分支，专注于从图像或视频中理解三维场景和对象的结构。
-   **分类 (Classification)**: 一种计算机视觉任务，旨在识别图像中存在的主要对象类别。
-   **语义分割 (Semantic Segmentation)**: 一种计算机视觉任务，为图像中的每个像素分配一个类别标签（如“草地”、“猫”、“树”），而不区分同一类别的不同实例。
-   **目标检测 (Object Detection)**: 一种计算机视觉任务，旨在识别图像中特定对象的实例，并用边界框定位它们，同时给出其类别。
-   **实例分割 (Instance Segmentation)**: 一种计算机视觉任务，结合了目标检测和语义分割，不仅识别和定位每个对象实例，还为每个实例提供精确的像素级掩码。
-   **深度图 (Depth Map)**: 一种图像表示，其中每个像素的值代表从相机到该像素处对应世界点（对象）的距离。
-   **RGB-D图像 (RGB-D Image)**: 结合了彩色（RGB）图像和深度图（D）的图像格式，常用于表示2.5维数据。
-   **尺度/深度模糊性 (Scale / Depth Ambiguity)**: 在单张图像中，由于透视投影，小而近的物体可能与大而远的物体在视觉上呈现出相同的大小和形状，导致其绝对尺度和深度无法确定。
-   **尺度不变损失 (Scale invariant loss)**: 一种损失函数，在训练神经网络预测深度图时，不惩罚预测深度与真实深度之间的全局尺度偏差，以解决尺度/深度模糊性问题。
-   **表面法线 (Surface Normals)**: 在3D几何中，与表面正交的向量，用于描述表面的方向和局部几何形状。
-   **体素网格 (Voxel Grid)**: 一种三维离散表示，将三维空间划分为小的立方体单元（体素），每个体素可以表示是否被物体占用（0或1），类似于2D图像中的像素网格。
-   **3D卷积 (3D Convolution)**: 卷积操作在三维空间中的扩展，使用三维卷积核在三维输入数据（如体素网格）上进行滑动和特征提取。
-   **八叉树 (Oct-Trees)**: 一种分层数据结构，用于高效地表示和存储稀疏的3D体素数据，通过在不同区域使用不同的分辨率来减少内存消耗。
-   **嵌套形状层 (Nested Shape Layers)**: 一种将形状预测为正空间和负空间组合的方法，通过多层稀疏体素表示来捕捉不同尺度的形状信息。
-   **体素管 (Voxel Tubes)**: 一种利用2D卷积来间接生成3D体素表示的方法，通过将深度信息编码为特征通道。
-   **隐式函数 (Implicit Function)**: 一种数学函数，其零值集（或某个特定水平集）定义了几何形状的表面，可以学习来分类三维空间中的点是位于形状内部还是外部。
-   **符号距离函数 (Signed Distance Function, SDF)**: 一种特殊的隐式函数，其值表示空间中任意点到最近表面点的欧几里得距离，其符号表示该点位于表面内部（负值）还是外部（正值）。
-   **点云 (Point Cloud)**: 由三维空间中的一系列点组成的数据集，用于表示物体的表面或空间结构。
-   **Chamfer 距离 (Chamfer distance)**: 一种用于衡量两个点集之间相似度的度量，计算一个集合中每个点到另一个集合中最近点的距离之和。
-   **F1 分数 (F1 Score)**: 在分类任务中常用的度量，是精确度 (Precision) 和召回率 (Recall) 的调和平均值，可以衡量模型的性能。
-   **网格 (Mesh)**: 一种由顶点、边和面（通常是三角形或四边形）组成的3D几何表示，用于描述物体的表面。
-   **迭代细化 (Iterative Refinement)**: 一种逐步改进模型或预测结果的方法，通常从一个粗略的初始猜测开始，通过多次迭代使其逐渐接近最终解。
-   **图卷积 (Graph Convolution)**: 一种用于在图结构数据上进行特征学习的卷积操作，通过聚合相邻节点的特征来更新当前节点的特征。
-   **顶点对齐特征 (Vertex-Aligned Features)**: 通过将3D顶点投影到图像平面，并从图像特征图中采样对应位置的特征，从而将图像特征与3D顶点对齐。
-   **Mesh Regularizer (网格正则化器)**: 在训练神经网络生成网格时，用于鼓励生成平滑、非退化网格的额外损失项，例如限制网格边的长度或网格面的面积。
-   **规范坐标 (Canonical Coordinates)**: 一种标准化的3D坐标系统，用于统一表示某一类物体（例如，所有椅子都朝向同一方向），独立于其在场景中的实际位置和姿态。
-   **视图坐标 (View Coordinates)**: 一种3D坐标系统，其原点和轴向与相机对齐，表示物体相对于相机的姿态。
-   **Amodal 补全 (Amodal Completion)**: 预测物体完整的三维形状，包括被其他物体遮挡（不可见）的部分。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **预测深度图 (Predicting Depth Maps)**:
    -   架构: 全卷积网络 (Fully Convolutional network)
    -   损失函数: 逐像素L2距离损失 (Per-pixel Loss (L2 Distance))，尺度不变 (Scale invariant)。

-   **预测表面法线 (Predicting Normals)**:
    -   架构: 全卷积网络
    -   损失函数: 逐像素点积损失 (Per-pixel Loss: $(x \cdot y) / (|x| |y|)$)。

-   **处理体素输入：3D卷积 (Processing Voxel Inputs: 3D Convolution)**:
    -   架构: 3D ShapeNets。
    -   输入：体素网格 (1 x 30 x 30 x 30)。
    -   训练: 分类损失函数 (classification loss)。

-   **生成体素形状：3D卷积 (Generating Voxel Shapes: 3D Convolution)**:
    -   任务：输入2D图像，输出3D体素网格。
    -   架构：2D CNN -> 3D Features -> 3D CNN。
    -   训练: 逐体素交叉熵损失 (per-voxel cross-entropy loss)。

-   **隐式函数：从隐式表示中提取形状 (Implicit Functions: Extracting Shapes from Implicit Representations)**:
    -   训练方式：学习神经网络，将3D坐标作为输入，输出该点是位于形状内部还是外部的概率。
    -   提取显式形状输出: 需要后处理（如Marching Cubes算法）。

-   **处理点云输入：PointNet (Processing Pointcloud Inputs: PointNet)**:
    -   目标：将点云作为集合处理，点的顺序不应影响结果。
    -   架构：输入点云 (P x 3) -> 对每个点运行MLP -> Max-Pool -> 全连接层。

-   **生成点云输出 (Generating Pointcloud Outputs)**:
    -   任务：输入单张RGB图像，输出点云。
    -   架构：2D CNN -> 图像特征 -> 两个分支：全连接分支和卷积分支，最后合并点云。

-   **预测点云：损失函数 (Predicting Point Clouds: Loss Function)**:
    -   **Chamfer 距离 (Chamfer distance)**:
        -   公式: $d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2^2 + \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2^2$
    -   **F1 分数 (F1 Score)**:
        -   F1@t = $2 \times (\text{Precision@t} \times \text{Recall@t}) / (\text{Precision@t} + \text{Recall@t})$

-   **预测网格：Pixel2Mesh (Predicting Meshes: Pixel2Mesh)**:
    -   输入：单张RGB图像。输出：对象的三角形网格。
    -   **核心思想 (Key Ideas)**:
        1.  **迭代细化 (Iterative Refinement)**: 从初始椭球体网格开始，网络预测每个顶点的偏移量，并重复该过程细化网格。
        2.  **图卷积 (Graph Convolution)**: 新的顶点特征依赖于自身特征和相邻顶点特征。
        3.  **顶点对齐特征 (Vertex-Aligned Features)**: 将3D顶点投影到图像平面，从2D CNN的特征图中采样，将图像信息整合到3D网格中。
        4.  **Chamfer 损失函数 (Chamfer Loss Function)**: 用于比较预测网格和真实网格的损失。

-   **Mesh R-CNN 流程 (Pipeline)**:
    1.  2D物体识别 (Mask R-CNN): 从输入图像获取边界框和分割掩码。
    2.  3D物体体素: 从2D预测生成粗略的3D体素。
    3.  3D物体网格: 将体素转换为网格，然后通过迭代细化获得高精度的网格预测。
    -   **Mesh Regularizers (形状正则化器)**:
        -   $L_{edge}$ = 最小化预测网格中边的L2范数，使网格更平滑。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   关于3D计算机视觉的总体问题 (Any questions on the sort of preamble about 3D computer vision before we really really dive into these different types of models?) [04:30]
-   为什么输入是30 x 30 x 30，然后只有1个通道 (Why does the input has a 30 x 30 x 30 and then only one channel?) [18:35]
-   卷积核是否是二进制的 (Does the kernel that operates into this binary...?) [20:36]
-   当我们从3D卷积模型转向体素管模型时，我们会牺牲什么 (what do we sort of sacrifice when we move from this 3D convolution model to this voxel tube representation model?) [26:26]
-   这些不同的方法清晰吗 (is is maybe this these these two different approaches of 3D convolution and voxel to representations clear for predicting voxel outputs?) [26:20]

---
1.什么是图卷积
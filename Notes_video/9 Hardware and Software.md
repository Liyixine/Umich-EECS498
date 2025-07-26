### [📚] 视频学习脚手架: Deep Learning Hardware and Software

### 一、核心内容大纲 (Core Content Outline)

-   **引言 (Introduction)**
    -   今日主题：深度学习硬件与软件 (Deep Learning Hardware and Software).
    -   课程性质：应用和实践，将涉及代码演示 (Applied and practical, with code examples).
-   **回顾：卷积神经网络架构 (Recall: CNN Architectures)**
    -   CNN 架构的发展历程 (Evolution of CNN architectures): 从 AlexNet 到 VGG，再到 ResNet.
    -   不同架构在计算和内存方面的权衡 (Trade-offs in computation and memory for different architectures).
    -   VGG 和 ResNet 提供了规则化的设计，易于扩展 (VGG and ResNet offered regular designs for easy scaling).
-   **深度学习硬件 (Deep Learning Hardware)**
    -   **计算机内部组件 (Inside a computer)**
        -   中央处理器 (CPU: Central Processing Unit): 通常位于主板顶部散热器下方。
        -   图形处理器 (GPU: Graphics Processing Unit): 物理尺寸远大于 CPU，占据更多空间。
    -   **NVIDIA 对比 AMD (NVIDIA vs AMD)**
        -   在游戏领域两者存在竞争和偏好 (Competition in gaming community).
        -   在深度学习领域，NVIDIA 是明显的赢家 (NVIDIA is the clear winner in deep learning).
        -   原因：NVIDIA 的软件栈 (CUDA) 更为先进，AMD 的硬件虽好但软件支持不足 (NVIDIA's software stack (CUDA) is more advanced, AMD's hardware lacks sufficient software support).
        -   这意味着深度学习中的 GPU 几乎特指 NVIDIA GPU (GPU in deep learning almost exclusively means NVIDIA GPUs).
    -   **GigaFLOPS/美元趋势 (GigaFLOPS per Dollar Trend)**
        -   GigaFLOPS 指每秒十亿次浮点运算 (GigaFLOPS: billions of floating-point operations per second).
        -   图表显示，CPU 和 GPU 的 FLOPs/美元都在随时间增长 (Both CPU and GPU FLOPS/dollar have been increasing over time).
        -   自 2012 年左右，GPU 在计算成本上出现爆炸性增长（即成本大幅下降）(Dramatic reduction in computing cost for GPUs since around 2012).
        -   NVIDIA GeForce 8800 GTX 是首款支持 CUDA 的显卡 (GeForce 8800 GTX was the first NVIDIA graphics card to support CUDA).
        -   AlexNet 使用的 GTX 580 极大地降低了计算成本，使得大型 CNNs 成为可能 (GTX 580 for AlexNet made large CNNs feasible due to lower computing cost).
        -   GPU 算力持续加速增长，摩尔定律并未失效 (GPUs continue to accelerate, Moore's Law doesn't seem dead for them).
    -   **CPU 与 GPU 对比 (CPU vs GPU Comparison)**
        -   **CPU (例如 Ryzen 9 3950X)**:
            -   更少的核心数 (Fewer cores: 16 cores, 32 threads).
            -   更高的主频 (Higher clock speed: 3.5 GHz base, 4.7 GHz boost).
            -   使用系统内存 (System RAM).
            -   价格：约 $749.
            -   算力：约 4.8 TFLOPs (FP32).
            -   特点：每个核心更强大、更快，擅长顺序任务 (Fewer but faster and more capable cores, great for sequential tasks).
        -   **GPU (例如 NVIDIA Titan RTX)**:
            -   更多核心数 (More cores: 4608 cores).
            -   较低主频 (Lower clock speed: 1.35 GHz base, 1.77 GHz boost).
            -   自带高速显存 (Dedicated memory: 24 GB GDDR6).
            -   价格：约 $2499.
            -   算力：约 16.3 TFLOPs (FP32)，使用 Tensor Cores 可达 130 TFLOPs.
            -   特点：更多核心，但每个核心较慢且“简单”，擅长并行任务 (More cores, but each core is slower and "dumber," great for parallel tasks).
    -   **GPU 内部结构：RTX Titan (Inside a GPU: RTX Titan)**
        -   GPU 像一个微型计算机，有自己的风扇、内存和处理器 (GPU is like a mini-computer with its own fans, memory, and processor).
        -   包含 12 个 2GB 内存模块 (12x 2GB memory modules).
        -   核心计算单元是流式多处理器 (SM: Streaming Multiprocessors): RTX Titan 有 72 个 SMs.
        -   每个 SM 包含 64 个 FP32 核心 (Each SM has 64 FP32 cores).
        -   **Tensor Cores**: 专门为深度学习设计，进行混合精度计算（FP16 乘法，FP32 加法）(Specialized hardware for mixed precision: FP16 multiplication and FP32 addition).
            -   一个 Tensor Core 能在一个时钟周期内计算 4x4 矩阵的 (A\*B + C) 运算 (128 FLOPs).
            -   Tensor Cores 使得 GPU 算力显著提升近 10 倍 (Tensor Cores offer a nearly 10x speedup over FP32 cores).
    -   **GPU 编程 (Programming GPUs)**
        -   **CUDA**: NVIDIA 专用语言，允许编写直接运行在 GPU 上的 C-like 代码 (NVIDIA-only, C-like code runs directly on GPU).
            -   NVIDIA 提供优化的 API，如 cuBLAS、cuFFT、cuDNN 等 (NVIDIA provides optimized APIs).
            -   通常深度学习实践者无需直接编写 CUDA (Practitioners usually don't need to write CUDA directly).
        -   **OpenCL**: 类似于 CUDA，但可在任何硬件上运行，通常在 NVIDIA 硬件上较慢 (Similar to CUDA, runs on anything, usually slower on NVIDIA hardware).
    -   **算力扩展 (Scaling Up Compute)**
        -   通常一个服务器包含 8 个 GPU (Typically 8 GPUs per server, e.g., NVIDIA DGX-1).
        -   **Google Tensor Processing Units (TPU)**: 谷歌推出的专用硬件 (Google's specialized hardware).
            -   TPU v2 (Cloud TPU v2): 180 TFLOPs，64 GB HBM 内存，可租赁 ($4.50/小时) 或在 Colab 上免费使用 (180 TFLOPs, 64 GB HBM memory, rentable or free on Colab).
            -   TPU v2 Pod: 包含 64 个 TPU-v2，算力达 11.5 PFLOPs (64 TPU-v2 devices, 11.5 PFLOPs).
            -   TPU v3 (Cloud TPU v3): 更强大的版本，420 TFLOPs，128 GB HBM 内存，可租赁 ($8/小时) (More powerful, 420 TFLOPs, 128 GB HBM memory, rentable).
            -   TPU v3 Pod: 包含 256 个 TPU-v3，算力达 107 PFLOPs (256 TPU-v3 devices, 107 PFLOPs).
            -   TPU 必须与 TensorFlow 框架配合使用 (TPUs require TensorFlow). (但未来可能支持 PyTorch, GitHub 上已有相关 Pull Request).
            -   算力与内存：计算型 GPU 比消费级 GPU 有更多内存和更高的内存带宽 (Compute-oriented GPUs have more memory and higher memory bandwidth than consumer GPUs).

-   **深度学习软件 (Deep Learning Software)**
    -   **框架的演变 (Evolution of frameworks)**
        -   早期学院派框架 (Early academic frameworks): Caffe, Torch, Theano.
        -   由大型公司维护的第二、三代框架 (Second and third-generation frameworks maintained by large companies): Caffe2 (Facebook), PyTorch (Facebook), TensorFlow (Google), MXNet (Amazon), CNTK (Microsoft), PaddlePaddle (Baidu), JAX (Google).
    -   **深度学习框架的目标 (Points of Deep Learning Frameworks)**:
        1.  实现新想法的快速原型开发 (Allow rapid prototyping of new ideas).
        2.  自动计算梯度 (Automatically compute gradients).
        3.  在 GPU（或 TPU）上高效运行 (Run efficiently on GPU (or TPU)).
    -   **PyTorch 框架 (PyTorch Framework)**
        -   **版本 (Versions)**: 本课程使用 PyTorch 1.2 版 (Released August 2019)。请注意旧版本 (0.3 到 0.4) 有重大 API 变化。Colab 环境可能在无通知情况下更新 PyTorch 版本，这可能影响随机种子输出。
        -   **核心概念 (Fundamental Concepts)**:
            -   **张量 (Tensor)**: 类似于 NumPy 数组，但可以在 GPU 上运行 (Like a NumPy array, but can run on GPU)。(在作业 A1-A3 中使用)
            -   **Autograd**: 用于从张量构建计算图并自动计算梯度的包 (Package for building computational graphs out of Tensors and automatically computing gradients)。 (在作业 A4-A6 中使用)
            -   **模块 (Module)**: 一个神经网络层；可能存储状态或可学习权重 (A neural network layer; may store state or learnable weights)。 (在作业 A4-A6 中使用)
        -   **PyTorch: 张量操作 (PyTorch: Tensors)**:
            -   用于在 CPU/GPU 上创建随机张量和执行基本操作。
        -   **PyTorch: Autograd (自动求导)**:
            -   通过设置 `requires_grad=True` 启用自动求导功能。
            -   对带有 `requires_grad=True` 的张量进行操作会使 PyTorch 动态构建计算图。
            -   `loss.backward()`: 自动计算损失对所有需要梯度的输入的梯度。
            -   **重要**: 每次梯度计算后需要将梯度清零 (`.zero_()`)，否则会累积。
            -   `with torch.no_grad():`: 告诉 PyTorch 在此上下文管理器中的操作不构建计算图，通常用于权重更新。
        -   **PyTorch: 定义新函数 (New functions)**:
            -   可以使用 Python 函数定义新的操作，其内部的 PyTorch 操作会添加到计算图中。
            -   **挑战**: 某些函数的梯度计算在数值上可能不稳定 (e.g., Sigmoid 的朴素实现)。
            -   **解决方案**: 通过继承 `torch.autograd.Function` 定义新的 `autograd` 操作，实现自定义的前向 (`forward`) 和后向 (`backward`) 传播，以提供数值稳定的梯度计算。
        -   **PyTorch: nn (神经网络模块)**:
            -   高级封装，用于处理神经网络，简化模型构建。
            -   `torch.nn.Sequential`: 对象导向的 API，将模型定义为层 (modules) 的堆叠。
            -   `torch.nn.functional`: 包含常用的辅助函数，如损失函数 (`mse_loss`)。
            -   `torch.optim`: 提供优化器对象，实现不同的梯度更新规则 (e.g., Adam, SGD)。
            -   **自定义模块 (Defining Modules)**: 非常常见，通过继承 `torch.nn.Module` 定义自己的模型或层。
                -   `__init__` 方法初始化子模块，模块可以包含其他模块。
                -   `forward` 方法定义前向传播，使用子模块和张量操作。
                -   无需定义 `backward`，Autograd 会自动处理。
            -   可以混合使用自定义模块子类和 `Sequential` 容器。
        -   **PyTorch: 数据加载器 (DataLoaders)**:
            -   `DataLoader` 包装 `Dataset` (如 `TensorDataset`)，提供小批量处理 (minibatching)、数据混洗 (shuffling)、多线程加载 (multithreading) 等功能。
        -   **PyTorch: 预训练模型 (Pretrained Models)**:
            -   通过 `torchvision` 可以非常容易地使用预训练模型。
    -   **动态计算图 (Dynamic Computation Graphs) 的应用 (Applications)**:
        -   模型结构依赖于输入 (Model structure depends on the input)。
        -   循环神经网络 (Recurrent Networks): 计算图的展开长度取决于输入序列的长度。
        -   递归神经网络 (Recursive Networks): 计算图的结构取决于输入的句法树等递归结构。
        -   模块化网络 (Modular Networks): 模型的一部分生成一个“程序”，另一部分根据这个程序执行计算。
    -   **静态图 vs 动态图：优化 (Static vs Dynamic Graphs: Optimization)**:
        -   静态图允许框架在运行前进行图优化，例如融合操作 (fused operations) 以提高效率。
    -   **静态图 vs 动态图：序列化 (Static vs Dynamic Graphs: Serialization)**:
        -   **静态图 (Static)**: 一旦图构建完成，就可以序列化并运行，无需构建图的代码。例如：在 Python 中训练模型，部署在 C++ 中。
        -   **动态图 (Dynamic)**: 图的构建和执行是交织的，因此总是需要保留构建图的代码。
    -   **静态图 vs 动态图：调试 (Static vs Dynamic Graphs: Debugging)**:
        -   **静态图 (Static)**: 代码编写和运行之间有很多间接性，调试、基准测试等可能很困难。
        -   **动态图 (Dynamic)**: 你编写的代码就是运行的代码，易于理解、调试、分析。
    -   **TensorFlow 版本和演进 (TensorFlow Versions and Evolution)**:
        -   **TensorFlow 1.0**: 默认使用静态图 (Default: static graphs)。
            -   API 复杂，调试困难，有时混乱 (Can be confusing to debug, API a bit messy)。
            -   代码范式：先定义计算图 (使用 `tf.placeholder` 等)，然后在一个 `tf.Session` 中运行。
        -   **TensorFlow 2.0**: 默认使用动态图 (Default: dynamic graphs)。
            -   API 更为简洁，与 PyTorch 的命令式风格相似。
            -   权重需要包装在 `tf.Variable` 中，以便能够修改它们。
            -   `tf.GradientTape`: 作用域前向传播，告诉 TensorFlow 开始构建计算图以便计算梯度。
            -   `tape.gradient()`: 请求 Tape 计算损失对参数的梯度。
            -   **`@tf.function`**: 类似于 PyTorch 的 `torch.jit.script`，将 Python 函数编译成静态图，以获得性能优势。
                -   编译后的图可以包含梯度计算和更新。
            -   **Keras**: TensorFlow 2.0 的高级 API 已标准化为 Keras。提供面向对象的模型构建方式（如 `Sequential`），以及常见的损失函数和优化器。
        -   **TensorBoard**: Google 开发的可视化工具，可记录损失、统计数据等，并在 Web 界面中显示精美图表。TensorFlow 和 PyTorch 都支持 (`torch.utils.tensorboard`)。

### 二、关键术语定义 (Key Term Definitions)

-   **中央处理器 (CPU: Central Processing Unit)**: 计算机的主要处理器，擅长执行顺序任务，核心数量相对较少但每个核心功能强大。
-   **图形处理器 (GPU: Graphics Processing Unit)**: 专为并行计算设计，拥有大量核心，每个核心相对简单但总体算力强大，在深度学习中广泛应用。
-   **GigaFLOPS (GFLOPS)**: 每秒十亿次浮点运算 (Billions of Floating Point Operations Per Second)，衡量计算设备算力的单位。
-   **Tensor Cores**: NVIDIA GPU 中用于深度学习的特殊硬件单元，擅长执行混合精度（FP16 乘法，FP32 加法）的矩阵乘法运算，极大地提高了深度学习的计算效率。
-   **流式多处理器 (SM: Streaming Multiprocessor)**: NVIDIA GPU 中的核心计算单元，包含多个 FP32 核心和 Tensor Cores。
-   **计算图 (Computational Graph)**: 深度学习框架的核心抽象，表示模型计算过程的有向无环图，用于自动微分和高效执行。
-   **张量 (Tensor)**: PyTorch 和 TensorFlow 等框架中的基本数据结构，类似于多维数组 (NumPy array)，但具备在 GPU 上运行和自动求导的能力。
-   **Autograd (自动求导)**: PyTorch 中的一个子系统，负责在计算过程中自动构建计算图并计算梯度。
-   **模块 (Module)**: PyTorch 中对神经网络层或整个模型的抽象，可以包含可学习的参数（权重）和子模块，便于构建复杂的网络结构。
-   **谷歌张量处理单元 (TPU: Tensor Processing Unit)**: 谷歌为加速机器学习工作负载而开发的专用集成电路 (ASIC)，针对神经网络的特定计算需求进行优化，通常与 TensorFlow 框架结合使用。
-   **混合精度 (Mixed Precision)**: 在深度学习中同时使用 FP16（半精度浮点数）和 FP32（单精度浮点数）进行计算的技术，以在保持模型准确性、减少内存使用量的同时提高计算速度。
-   **高带宽内存 (HBM: High Bandwidth Memory)**: 一种高性能 RAM 接口，通常用于计算型 GPU 和加速器，提供极高的内存带宽。
-   **动态计算图 (Dynamic Computation Graphs)**: 计算图在每次前向传播时根据代码的执行路径动态构建，支持条件判断和循环等控制流，便于调试。
-   **静态计算图 (Static Computation Graphs)**: 计算图在训练开始前一次性构建并固定，框架可以对其进行编译和优化，以提高运行效率和部署能力。
-   **JIT (Just-In-Time Compilation)**: 即时编译，一种在程序运行时才进行编译的技术，在深度学习框架中用于将动态图转换为静态图以优化性能。
-   **Keras**: 一个高级神经网络 API，可以在 TensorFlow、Theano 或 CNTK 之上运行。在 TensorFlow 2.0 中，Keras 被集成并作为其官方推荐的高级 API。
-   **TensorBoard**: Google 开发的可视化工具，用于在训练深度学习模型时跟踪和可视化各种指标，如损失、准确率、权重分布等。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **PyTorch: 手动两层 ReLU 网络训练 (Manual Two-Layer ReLU Network Training in PyTorch)**:
    该代码示例展示了如何使用 PyTorch 的 Tensor API 从头实现一个两层 ReLU 网络的训练过程，包括前向传播、损失计算、手动计算梯度和参数更新。
    ```python
    import torch

    device = torch.device('cpu') # 可以改为'cuda'在GPU上运行

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    w1 = torch.randn(D_in, H, device=device)
    w2 = torch.randn(H, D_out, device=device)

    learning_rate = 1e-6

    for t in range(500):
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 0:
            print(t, loss.item())

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    ```

-   **PyTorch: 使用 Autograd 自动求导 (PyTorch: Autograd for Automatic Differentiation)**:
    该片段展示了如何使用 `requires_grad=True` 和 `loss.backward()` 来自动计算梯度，以及使用 `with torch.no_grad():` 来禁用图构建。
    ```python
    import torch

    device = torch.device('cuda') # Running on GPU

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    w1 = torch.randn(D_in, H, device=device, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, requires_grad=True)

    learning_rate = 1e-6

    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 0:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()
    ```

-   **PyTorch: 定义自定义 Autograd 操作 (PyTorch: Defining Custom Autograd Operators)**:
    该片段展示了如何为 Sigmoid 函数定义一个自定义的 `torch.autograd.Function`，以提供数值稳定的反向传播。
    ```python
    class Sigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = 1.0 / (1.0 + (-x).exp())
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            y = ctx.saved_tensors[0]
            grad_x = grad_y * y * (1.0 - y)
            return grad_x

    # Usage in training loop:
    # y_pred = Sigmoid.apply(x.mm(w1))
    ```

-   **PyTorch: 使用 nn.Module 构建模型 (PyTorch: Building Models with nn.Module)**:
    该片段展示了如何使用 `torch.nn.Sequential` 构建一个简单的两层网络，或定义一个自定义的 `nn.Module` 子类。
    ```python
    import torch
    import torch.nn as nn

    # Example with Sequential
    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out)
    )

    # Example with custom Module subclass
    class TwoLayerNet(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = nn.Linear(D_in, H)
            self.linear2 = nn.Linear(H, D_out)

        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    model = TwoLayerNet(D_in, H, D_out)

    # Training loop with nn.Module and optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for t in range(500):
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    ```

### 四、讲师提出的思考题 (Questions Posed by the Instructor)

-   深度学习框架的主要目的是什么？
-   为什么在 PyTorch 中使用 `requires_grad=True` 和 `loss.backward()` 可以自动计算梯度？
-   为什么每次梯度计算后需要将梯度清零 (`.zero_()`)？
-   PyTorch 中自定义函数和自定义 `autograd.Function` 有何区别？何时选择哪种方式？
-   静态图和动态图在优化和部署方面有何优劣？
-   PyTorch 和 TensorFlow 之间现在的主要异同点是什么？

---
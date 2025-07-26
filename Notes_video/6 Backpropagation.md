### [📚] 视频学习脚手架: CS231n Lecture 6: 神经网络与反向传播 (Neural Networks & Backpropagation)

### 一、核心内容大纲 (Core Content Outline)
-   **上节课回顾: 神经网络 (Last time: Neural Networks)**
    -   从线性分类器到全连接网络
    -   核心概念:
        -   空间扭曲 (Space Warping)
        -   通用近似定理 (Universal Approximation)
        -   非凸性 (Nonconvex)
-   **核心问题: 如何计算梯度 (Problem: How to compute gradients?)**
    -   目标: 计算总损失 (Total loss) L 相对于权重 W1 和 W2 的偏导数。
    -   总损失 = 数据损失 (data loss) + 正则化 (regularization)
-   **（糟糕的）想法: 在纸上手动推导梯度 (Bad Idea: Derive ∇L on paper)**
    -   展开 SVM 损失函数的完整表达式。
    -   这种方法存在的问题:
        -   **问题1**: 非常繁琐，需要大量矩阵微积分和纸张。
        -   **问题2**: 非模块化。如果想更换损失函数（如用 Softmax 替换 SVM），需要从头重新推导。
        -   **问题3**: 对于非常复杂的模型，此方法不可行。
-   **更好的想法: 计算图 (Better Idea: Computational Graphs)**
    -   将任意复杂函数表示为一个有向图。
    -   图中的节点代表基本运算或变量。
    -   示例:
        -   简单的线性分类器 (`f = Wx`) 与 Hinge Loss。
        -   复杂的深度网络，如 AlexNet。
        -   极其复杂的模型，如神经图灵机 (Neural Turing Machine)。
-   **反向传播 (Backpropagation): 在计算图上计算梯度**
    -   **简单标量示例: `f(x, y, z) = (x + y)z`**
        -   **前向传播 (Forward pass)**: 从输入到输出，计算每个节点的输出值。
        -   **后向传播 (Backward pass)**: 从最终输出开始，反向计算梯度。
            -   利用链式法则 (Chain Rule)。
            -   引入概念: 上游梯度 (Upstream Gradient), 局部梯度 (Local Gradient), 下游梯度 (Downstream Gradient)。
            -   **下游梯度 = 局部梯度 × 上游梯度**
    -   **通用化: 模块化视角**
        -   将每个运算节点视为一个独立的模块 (gate)。
        -   每个节点接收上游梯度，并根据其局部梯度计算下游梯度。
-   **梯度流中的模式 (Patterns in Gradient Flow)**
    -   **加法门 (add gate)**: 扮演“梯度分配器” (gradient distributor) 的角色，将上游梯度等值地分配给所有输入。
    -   **拷贝门 (copy gate)**: 扮演“梯度加法器” (gradient adder) 的角色，将来自不同分支的梯度相加。
    -   **乘法门 (mul gate)**: 扮演“交换乘法器” (swap multiplier) 的角色，一个输入的梯度是上游梯度乘以另一个输入的值。
    -   **最大值门 (max gate)**: 扮演“梯度路由器” (gradient router) 的角色，将梯度只路由到值较大的那个输入，另一个输入的梯度为0。
-   **处理向量 (Backprop with Vectors)**
    -   回顾向量导数:
        -   标量对向量求导 -> 梯度 (Gradient)
        -   向量对向量求导 -> 雅可比矩阵 (Jacobian Matrix)
    -   雅可比矩阵通常是稀疏的 (sparse)，尤其是对于元素级 (elementwise) 操作（此时为对角矩阵）。
    -   **核心思想**: 永远不要显式地构建巨大的雅可比矩阵，而是利用其结构进行隐式乘法 (implicit multiplication)。
    -   示例: 对 ReLU 函数进行向量化的反向传播。
-   **处理矩阵或张量 (Backprop with Matrices or Tensors)**
    -   概念与向量类似，但雅可比矩阵会变成更高阶的张量。
    -   **关键**: 将高阶张量运算“扁平化” (flatten) 为矩阵-向量乘法来理解。
    -   示例: 矩阵乘法 `y = xw` 的反向传播推导。
        -   通过分析单个元素的梯度来推导整体的梯度表达式。
        -   最终的梯度计算可以简化为两个矩阵乘法。
-   **实现策略 (Implementation Strategies)**
    -   **“扁平化”反向传播 (Flat Backprop)**
        -   将前向传播的代码“反向”过来写。
        -   适用于固定结构的简单模型，例如作业2中的要求。
        -   缺点: 非模块化，修改模型结构需要重写大量代码。
    -   **模块化 API (Modular API)**
        -   将计算图和节点都实现为对象 (object)。
        -   每个节点对象实现自己的 `forward` 和 `backward` 方法。
        -   图对象负责按拓扑顺序调用节点的 `forward` 和 `backward` 方法。
        -   这是现代深度学习框架（如 PyTorch）的实现方式。
    -   **PyTorch Autograd 示例**:
        -   通过继承 `torch.autograd.Function` 来定义自己的运算。
        -   实现静态的 `forward` 和 `backward` 方法。

### 二、关键术语定义 (Key Term Definitions)
-   **计算图 (Computational Graph)**: 一种将任意数学表达式表示为有向图的数据结构，其中节点代表变量或操作，边代表函数关系。
-   **反向传播 (Backpropagation)**: 一种在计算图上高效计算梯度的算法，通过递归地应用链式法则，从最终输出开始反向传播梯度。
-   **前向传播 (Forward Pass)**: 在计算图中，从输入开始，按拓扑顺序计算并存储每个节点的输出值的过程。
-   **后向传播 (Backward Pass)**: 在计算图中，从最终输出（其梯度为1）开始，按拓扑顺序的逆序，计算每个节点相对于最终输出的梯度的过程。
-   **上游梯度 (Upstream Gradient)**: 在反向传播中，一个节点的输出相对于最终损失函数的梯度（即从图的下游传来的梯度）。
-   **局部梯度 (Local Gradient)**: 一个节点的输出相对于其直接输入的偏导数。
-   **下游梯度 (Downstream Gradient)**: 在反向传播中，一个节点的输入相对于最终损失函数的梯度（即需要继续向下游传播的梯度）。
-   **雅可比矩阵 (Jacobian Matrix)**: 一个向量值函数相对于其向量输入的偏导数矩阵，概括了每个输入元素对每个输出元素的影响。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)
-   **算法: “扁平化”反向传播 (Flat Backprop)**
    1.  **前向传播**: 编写代码，计算从输入到最终输出L的所有中间变量。
        ```python
        # Forward pass: Compute output
        s0 = w0 * x0
        s1 = w1 * x1
        s2 = s0 + s1
        s3 = s2 + w2
        L = sigmoid(s3)
        ```
    2.  **后向传播**: 以相反的顺序，为前向传播中的每一行代码，根据链式法则计算对应的梯度。
        ```python
        # Backward pass: Compute grads
        grad_L = 1.0 # Base case
        grad_s3 = grad_L * (1 - L) * L # Backward through sigmoid
        grad_w2 = grad_s3 # Backward through add
        grad_s2 = grad_s3 # Backward through add
        grad_s0 = grad_s2 # Backward through add
        grad_s1 = grad_s2 # Backward through add
        grad_w1 = grad_s1 * x1 # Backward through multiply
        grad_x1 = grad_s1 * w1 # Backward through multiply
        grad_w0 = grad_s0 * x0 # Backward through multiply
        grad_x0 = grad_s0 * w0 # Backward through multiply
        ```
-   **代码示例: PyTorch 自定义 Autograd 函数 (PyTorch Autograd Functions)**
    ```python
    class Multiply(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            # Stash inputs for use in backward pass
            ctx.save_for_backward(x, y)
            z = x * y
            return z

        @staticmethod
        def backward(ctx, grad_z): # grad_z is the upstream gradient
            # Retrieve stashed inputs
            x, y = ctx.saved_tensors
            # Multiply upstream gradient by local gradients
            grad_x = y * grad_z # Local gradient for x is y
            grad_y = x * grad_z # Local gradient for y is x
            return grad_x, grad_y
    ```
-   **代码示例: PyTorch 底层 Sigmoid 层实现 (PyTorch Sigmoid Layer)**
    ```c
    // (Simplified from the lecture's C/C++ snippet)
    // Forward pass computes the sigmoid function
    void THNN_Sigmoid_updateOutput(THNNState *state, THTensor *input, THTensor *output) {
        // ...
        // output = 1 / (1 + exp(-input))
    }

    // Backward pass computes the gradient
    void THNN_Sigmoid_updateGradInput(THNNState *state, THTensor *gradOutput, THTensor *gradInput, THTensor *output) {
        // ...
        // gradInput = gradOutput * (1 - output) * output;
    }
    ```

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   `df/df` (f对f的导数) 应该是什么？ (答案: 1)
-   在一个线性模型中，我们实际上已经看到了这些（输入和权重）以两种方式被使用：一次是计算得分，另一次是计算正则化项。那么我们应该如何处理这种情况？(暗示：使用拷贝门)
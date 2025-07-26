### [📚] 视频学习脚手架: 优化 (Optimization)

### 一、核心内容大纲 (Core Content Outline)
-   **引言 (Introduction)**
    -   回顾 (Review):
        -   线性模型 (Linear Models) 用于图像分类问题 (Image Classification Problems)。
        -   损失函数 (Loss Functions) 用来表达对不同权重选择的偏好 (Express preferences over different choices of weights)。
    -   核心问题 (Core Problem): 如何找到使损失函数 $L(W)$ 最小化的权重矩阵 $W^* = \arg \min_W L(W)$。
    -   优化 (Optimization) 的直观理解 (Intuitive Understanding): 在多维损失景观中寻找最低点，类似蒙眼下山。

-   **优化策略 (Optimization Strategies)**
    -   **想法 #1: 随机搜索 (Random Search)**
        -   方法 (Method): 随机生成权重矩阵 $W$，计算损失，保留使损失最小的 $W$。
        -   评价 (Evaluation): 简单但效率极低，准确率很低，不适合实际应用。
    -   **想法 #2: 沿着坡度下降 (Follow the Slope)**
        -   核心 (Core): 沿着负梯度 (Negative Gradient) 方向移动，因为这是函数下降最快的方向 (最速下降 - Steepest Descent)。
        -   梯度类型 (Gradient Types):
            -   数值梯度 (Numeric Gradient): 通过有限差分近似计算，近似 (Approximate)、计算慢 (Slow)、易于编写 (Easy to write)。
            -   解析梯度 (Analytic Gradient): 通过微积分精确推导，精确 (Exact)、计算快 (Fast)、易于出错 (Error-prone)。
            -   实践中 (In Practice): 通常使用解析梯度，并用数值梯度进行梯度检查 (Gradient Check) 以验证正确性。

-   **梯度下降算法 (Gradient Descent Algorithm)**
    -   **香草梯度下降 (Vanilla Gradient Descent)** (全批量梯度下降 - Full Batch Gradient Descent)
        -   每次迭代 (Iteration) 使用所有训练数据计算梯度。
        -   问题 (Problem): 对于大型数据集，每次计算梯度成本过高。
    -   **随机梯度下降 (Stochastic Gradient Descent - SGD)**
        -   每次迭代从数据集中随机采样一个小的子集（小批量 - Minibatch）来计算梯度，以近似真实梯度。
        -   命名由来 (Nomenclature): 梯度是真实梯度的随机估计 (Stochastic Estimate)。
        -   超参数 (Hyperparameters): 权重初始化 (Weight Initialization)、迭代次数 (Number of Steps)、学习率 (Learning Rate)、批量大小 (Batch Size)、数据采样 (Data Sampling)。

-   **SGD 的常见问题 (Problems with SGD)**
    -   **问题1: 损失函数条件数高 (High Condition Number in Loss Function)**
        -   场景 (Scenario): 损失函数在某个方向上变化非常快（陡峭），而在另一个方向上变化非常慢（平坦）。Hessian 矩阵 (Hessian Matrix) 的最大奇异值与最小奇异值之比（条件数）很大。
        -   表现 (Behavior): 优化路径在陡峭方向上会发生剧烈震荡 (Jitter along steep direction)，而在平坦方向上进展非常缓慢 (Very slow progress along shallow dimension)。
        -   权衡 (Trade-off):
            -   学习率 (Learning Rate) 过大 (Too Big): 在陡峭方向上过度调整 (Overshoot)，导致优化路径来回锯齿形震荡 (Zigzagging Pattern)，需要更多步骤才能收敛。
            -   学习率过小 (Too Small): 可以避免过度调整，但会导致在平坦方向上收敛极其缓慢，几乎没有进展。
        -   根本原因 (Root Cause): 单一的学习率无法同时适应损失景观中不同方向的曲率。
    -   **问题2: 局部最小值 (Local Minimum) 或 鞍点 (Saddle Point)**
        -   局部最小值 (Local Minimum): 函数梯度为零的点，但并非全局最优解。梯度下降会卡在这些点。
        -   鞍点 (Saddle Point): 函数梯度也为零，但在一个方向上函数值增加，在另一个方向上函数值减少。在高维优化中，鞍点比局部最小值更常见。
        -   共同影响: 两种情况下，由于梯度为零 (Zero Gradient)，优化算法会停止更新 (Gradient Descent Gets Stuck)，无法进一步探索更优解。
    -   **问题3: 梯度噪声 (Noisy Gradients)**
        -   原因 (Reason): SGD 算法的梯度是从小批量样本中计算的，因此它是对完整数据集梯度的有噪声估计 (Noisy Estimate)。
        -   表现 (Behavior): 优化路径会显得不平滑，甚至会偏离最速下降方向，导致算法在损失景观中蜿蜒前行 (Meandering Around)，收敛速度减慢。

-   **SGD 的改进算法 (Improved SGD Algorithms)**
    -   **SGD + 动量 (SGD + Momentum)**
        -   直观理解 (Intuition): 引入“速度 (Velocity)”概念，作为梯度的运行平均值 (Running Mean of Gradients)，模拟一个有惯性的小球在损失景观中滚动。
        -   公式 (Formula):
            -   $v_{t+1} = \rho v_t + \nabla f(x_t)$ (更新速度，$\rho$ 是动量系数/摩擦力，通常取 0.9 或 0.99)
            -   $x_{t+1} = x_t - \alpha v_{t+1}$ (更新参数，$\alpha$ 是学习率)
        -   优点 (Advantages):
            -   克服高条件数 (High Condition Number): 累积速度，减少在陡峭方向上的震荡，加速在平坦方向上的进展。
            -   逃离局部最小值/鞍点 (Escape Local Minimum/Saddle Point): 小球的惯性 (Momentum) 帮助它越过零梯度的“山谷”或“鞍部”。
            -   平滑梯度噪声 (Smooth Gradient Noise): 通过平均梯度方向，使优化路径更平滑、更直接。
    -   **Nesterov 动量 (Nesterov Momentum)**
        -   直观理解 (Intuition): “展望 (Look Ahead)”一步，在根据当前速度预期的位置计算梯度，然后结合该梯度更新速度。相当于在到达当前点之前，提前预判下一步的梯度方向。
        -   公式 (Formula):
            -   $v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)$ (在“展望”点计算梯度并更新速度)
            -   $x_{t+1} = x_t + v_{t+1}$ (更新参数)
        -   效果 (Effect): 通常比标准动量收敛更快。
    -   **AdaGrad (自适应梯度 - Adaptive Gradient Algorithm)**
        -   核心思想 (Core Idea): 为每个参数维度维护历史梯度平方和 (Historical Sum of Squares)，并以此来元素级地 (Element-wise) 缩放学习率。
        -   伪代码 (Pseudocode):
            ```python
            grad_squared = 0  # Initialize element-wise to zeros
            for t in range(num_steps):
                dw = compute_gradient(w)
                grad_squared += dw * dw  # Accumulate squared gradients element-wise
                w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7) # Adaptive learning rate
            ```
        -   效果 (Effect): 对梯度变化快的维度（陡峭）减小学习率，对梯度变化慢的维度（平坦）增大学习率。
        -   问题 (Problem): 由于 `grad_squared` 持续累积且永不衰减，导致学习率在训练后期无限减小 (Learning Rate Approaches Zero)，可能使算法过早停滞 (Premature Stagnation)。
    -   **RMSProp (均方根传播 - Root Mean Square Propagation)**
        -   核心思想 (Core Idea): 解决 AdaGrad 学习率衰减过快问题，使用梯度平方的指数加权移动平均 (Exponentially Weighted Moving Average)，而非累积和。相当于“泄露 (Leak)”一部分历史信息。
        -   伪代码 (Pseudocode):
            ```python
            grad_squared = 0  # Initialize element-wise to zeros
            for t in range(num_steps):
                dw = compute_gradient(w)
                grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dw * dw # Exponentially weighted average
                w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
            ```
        -   效果 (Effect): 学习率更稳定，避免过早停滞，通常比 AdaGrad 表现更好。
    -   **Adam (自适应矩估计 - Adaptive Moment Estimation)**
        -   核心思想 (Core Idea): 结合动量 (Momentum) 和 RMSProp 的思想，同时对梯度的指数加权平均（一阶矩 - First Moment）和梯度平方的指数加权平均（二阶矩 - Second Moment）进行跟踪。
        -   伪代码 (Pseudocode):
            ```python
            moment1 = 0  # Initialize element-wise to zeros (first moment estimate)
            moment2 = 0  # Initialize element-wise to zeros (second moment estimate)
            for t in range(num_steps):
                dw = compute_gradient(w)
                moment1 = beta1 * moment1 + (1 - beta1) * dw
                moment2 = beta2 * moment2 + (1 - beta2) * dw * dw
                # Bias correction for initial steps
                moment1_unbias = moment1 / (1 - beta1**t)
                moment2_unbias = moment2 / (1 - beta2**t)
                w -= learning_rate * moment1_unbias / (moment2_unbias.sqrt() + 1e-7)
            ```
        -   关键改进 (Key Improvement): 引入偏差校正 (Bias Correction)，解决优化初期由于指数移动平均从零开始累积而导致的估计偏差，避免了初始步骤过大。
        -   实践中 (In Practice): Adam 是一种非常鲁棒且性能优秀的优化器，通常作为设计新深度学习模型时的首选 (Good Default Choice)。推荐超参数：`beta1=0.9`，`beta2=0.999`，`learning_rate` 在 $10^{-3}$ 到 $10^{-4}$ 之间。

-   **二阶优化 (Second-Order Optimization)**
    -   一阶优化 (First-Order Optimization): 使用梯度 (一阶导数 - First Derivative) 来构建损失函数的局部线性近似 (Local Linear Approximation)，然后沿着该近似的下降方向迈步。
    -   二阶优化 (Second-Order Optimization): 使用梯度和 Hessian 矩阵 (Hessian Matrix，二阶导数 - Second Derivative) 来构建损失函数的局部二次近似 (Local Quadratic Approximation)，然后直接跳到该二次近似的最小值点。
    -   牛顿参数更新 (Newton Parameter Update):
        -   损失函数 $L(w)$ 的二阶泰勒展开 (Second-Order Taylor Expansion):
            $L(w) \approx L(w_0) + (w-w_0)^T \nabla_w L(w_0) + \frac{1}{2}(w-w_0)^T H_w L(w_0) (w-w_0)$
        -   通过求解近似函数的临界点，得到牛顿参数更新公式:
            $w^* = w_0 - H_w L(w_0)^{-1} \nabla_w L(w_0)$
    -   **为什么在深度学习中不实用 (Why Impractical in Deep Learning)?**
        -   Hessian 矩阵的元素数量是 $O(N^2)$ (N 为参数数量)，对于深度学习模型 (参数数量通常为数百万甚至数亿)，存储成本极高。
        -   Hessian 矩阵求逆的计算复杂度是 $O(N^3)$，即使对于中等大小的模型也无法在合理时间内完成。
    -   **L-BFGS (限定内存 BFGS - Limited-memory BFGS)**
        -   一种拟牛顿 (Quasi-Newton) 方法，通过近似 Hessian 矩阵的逆来提高效率。
        -   通常在全批量 (Full Batch) 和确定性 (Deterministic) 模式下表现非常好，如果损失函数是确定性的且数据量不大，L-BFGS 可能是非常好的选择。
        -   不适用于小批量 (Mini-batch) 设置，因为梯度噪声会破坏其 Hessian 近似，导致性能不佳。将二阶方法应用于大规模、随机设置是活跃的研究领域。

-   **实践中的优化器选择总结 (Optimizer Choices in Practice Summary)**
    -   **Adam**: 在许多情况下是一个很好的默认选择。
    -   **SGD + 动量 (SGD + Momentum)**: 有时能超越 Adam，但可能需要更多超参数调整，特别是学习率和动量系数。
    -   **L-BFGS**: 如果能够进行全批量更新（例如，问题是确定性的或数据集足够小），L-BFGS 通常会表现非常好，但务必禁用所有形式的噪声。

### 二、关键术语定义 (Key Term Definitions)
-   **权重矩阵 (Weight Matrix)**: 线性分类器中的参数，通过学习来决定输入数据的分类方式。
-   **损失函数 (Loss Function)**: 量化模型预测结果与真实标签之间差异的函数，目标是最小化损失函数。
-   **正则化 (Regularization)**: 在损失函数中添加惩罚项，以避免模型过拟合 (Overfitting)。
-   **优化 (Optimization)**: 寻找使某个函数（如损失函数）达到最小值或最大值的过程。
-   **梯度 (Gradient)**: 多变量函数在某一点上的所有偏导数构成的向量，指向函数在该点增长最快的方向。
-   **负梯度 (Negative Gradient)**: 指向函数在该点下降最快的方向。
-   **数值梯度 (Numeric Gradient)**: 通过有限差分 (Finite Differences) 近似计算的梯度。
-   **解析梯度 (Analytic Gradient)**: 通过微积分推导出的梯度精确表达式。
-   **梯度检查 (Gradient Check)**: 对解析梯度实现正确性进行验证的方法，通常通过与数值梯度进行比较。
-   **梯度下降 (Gradient Descent)**: 一种迭代优化算法，沿着目标函数梯度的负方向更新参数，以找到函数的局部最小值。
-   **超参数 (Hyperparameters)**: 在机器学习模型训练之前设置的参数，而不是通过训练数据学习得到的参数。
-   **学习率 (Learning Rate)**: 梯度下降中控制每次参数更新步长大小的超参数。
-   **小批量 (Minibatch)**: 在 SGD 中，每次迭代时从训练数据集中随机抽取的一小部分样本。
-   **高条件数 (High Condition Number)**: 衡量损失函数表面曲率的指标，高条件数表示函数在不同方向上的变化率差异很大。
-   **Hessian 矩阵 (Hessian Matrix)**: 一个多变量实值函数的二阶偏导数组成的方阵，描述了函数局部曲率。
-   **局部最小值 (Local Minimum)**: 在函数定义域的某个子区域内，函数值最小的点。
-   **鞍点 (Saddle Point)**: 函数梯度为零，但在某些方向上函数值增加，在另一些方向上函数值减少的点。
-   **梯度噪声 (Gradient Noise)**: 由于在 SGD 中使用小批量样本计算梯度，导致梯度估计存在随机性或误差。
-   **动量 (Momentum)**: 优化算法中引入的一个概念，通过累积之前梯度的方向和大小，帮助优化器在平坦区域加速。
-   **Nesterov 动量 (Nesterov Momentum)**: 动量算法的一种变体，通过在计算梯度前“展望 (Look Ahead)”一步，获得更准确的梯度方向。
-   **自适应学习率 (Adaptive Learning Rates)**: 优化算法中的一类方法，根据每个参数的历史梯度信息，自动调整学习率。
-   **AdaGrad (Adaptive Gradient Algorithm)**: 一种自适应学习率优化算法，通过累积每个参数的梯度平方和来调整学习率。
-   **RMSProp (Root Mean Square Propagation)**: AdaGrad 的改进版本，通过使用梯度平方的指数加权移动平均来避免学习率过早衰减。
-   **Adam (Adaptive Moment Estimation)**: 结合了 Momentum 和 RMSProp 的思想，同时使用梯度的指数加权平均（一阶矩 - First Moment）和梯度平方的指数加权平均（二阶矩 - Second Moment）。
-   **偏差校正 (Bias Correction)**: 在 Adam 等算法中，用于修正优化初期由于指数移动平均从零开始累积而导致的估计偏差。
-   **一阶优化 (First-Order Optimization)**: 使用函数的一阶导数（梯度）来指导优化过程的算法。
-   **二阶优化 (Second-Order Optimization)**: 使用函数的二阶导数信息（如 Hessian 矩阵）来指导优化过程的算法。
-   **L-BFGS (Limited-memory BFGS)**: 一种拟牛顿 (Quasi-Newton) 优化算法，通过近似 Hessian 矩阵的逆来避免计算和存储整个 Hessian。
-   **全批量 (Full Batch)**: 在优化中，每次参数更新使用整个训练数据集计算梯度。
-   **确定性模式 (Deterministic Mode)**: 优化过程中不包含随机性，通常指使用全批量梯度或没有额外噪声。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **SGD (随机梯度下降) 算法**:
    $$x_{t+1} = x_t - \alpha \nabla f(x_t)$$
    其中 $x_t$ 是当前参数，$ \alpha $ 是学习率，$ \nabla f(x_t) $ 是损失函数在 $x_t$ 处的梯度。
    ```python
    # 伪代码
    # w: 权重参数
    # learning_rate: 学习率
    # num_steps: 训练步数

    for t in range(num_steps):
        dw = compute_gradient(w) # 计算小批量数据的梯度
        w -= learning_rate * dw # 更新权重
    ```

-   **SGD + 动量 (SGD + Momentum)**:
    $$v_{t+1} = \rho v_t + \nabla f(x_t)$$
    $$x_{t+1} = x_t - \alpha v_{t+1}$$
    其中 $v_t$ 是速度向量，$\rho$ 是动量系数 (或摩擦力)。
    ```python
    # 伪代码
    # v: 速度向量，初始化为0
    # rho: 动量系数 (例如 0.9 或 0.99)

    v = 0
    for t in range(num_steps):
        dw = compute_gradient(w) # 计算梯度
        v = rho * v + dw # 更新速度，累积梯度的方向和大小
        w -= learning_rate * v # 沿着速度方向更新权重
    ```

-   **Nesterov 动量 (Nesterov Momentum)**:
    $$v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)$$
    $$x_{t+1} = x_t + v_{t+1}$$
    特点是在计算梯度时，是计算“展望”一步后的梯度，而非当前点梯度。
    ```python
    # 伪代码
    # v: 速度向量，初始化为0
    # rho: 动量系数

    v = 0
    for t in range(num_steps):
        # 不同的实现形式，最终效果等价
        # 形式1:
        # x_ahead = w + rho * v # 展望一步
        # dw_ahead = compute_gradient(x_ahead) # 计算展望点的梯度
        # v = rho * v - learning_rate * dw_ahead
        # w += v

        # 形式2 (视频中展示的版本，便于与SGD+Momentum比较):
        dw = compute_gradient(w) # 计算当前w的梯度
        old_v = v # 保存旧的速度
        v = rho * v + dw # 更新速度 (注意这里没有负号，因为下面更新w时会减去)
        w += -learning_rate * dw # 先用当前梯度更新，再用旧速度更新 (效果等价于用新速度更新)

        # 视频中为了对比SGD+Momentum的更新公式，在Adam部分做了统一写法
        # Nesterov 的另一个等价形式：
        # dw = compute_gradient(w + rho * v) # 在预测位置计算梯度
        # v = rho * v - learning_rate * dw
        # w += v
    ```

-   **AdaGrad (自适应梯度 - Adaptive Gradient Algorithm)**:
    ```python
    # 伪代码
    # grad_squared: 梯度平方和，元素级，初始化为0

    grad_squared = 0
    for t in range(num_steps):
        dw = compute_gradient(w)
        grad_squared += dw * dw # 累积每个维度梯度的平方和
        # 学习率对每个维度进行自适应缩放，+1e-7防止除零
        w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
    ```

-   **RMSProp (均方根传播 - Root Mean Square Propagation)**:
    ```python
    # 伪代码
    # grad_squared: 梯度平方的指数加权移动平均，元素级，初始化为0
    # decay_rate: 衰减率 (例如 0.99)

    grad_squared = 0
    for t in range(num_steps):
        dw = compute_gradient(w)
        # 使用指数加权移动平均更新，避免无限累积
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dw * dw
        w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
    ```

-   **Adam (自适应矩估计 - Adaptive Moment Estimation)**:
    ```python
    # 伪代码
    # moment1: 梯度的指数加权平均 (一阶矩)，初始化为0
    # moment2: 梯度平方的指数加权平均 (二阶矩)，初始化为0
    # beta1, beta2: 衰减率 (例如 beta1=0.9, beta2=0.999)
    # learning_rate: 学习率

    moment1 = 0
    moment2 = 0
    for t in range(num_steps):
        dw = compute_gradient(w)

        # 更新一阶矩估计 (动量部分)
        moment1 = beta1 * moment1 + (1 - beta1) * dw
        # 更新二阶矩估计 (RMSProp部分)
        moment2 = beta2 * moment2 + (1 - beta2) * dw * dw

        # 偏差校正 (Bias Correction)
        # 修正初始化为零导致的矩估计在初期偏向零的问题
        moment1_unbias = moment1 / (1 - beta1**(t + 1)) # 注意这里通常是 t+1，因为t从0开始
        moment2_unbias = moment2 / (1 - beta2**(t + 1))

        # 结合一阶和二阶矩进行参数更新
        w -= learning_rate * moment1_unbias / (moment2_unbias.sqrt() + 1e-7)
    ```

-   **L-BFGS (限定内存 BFGS - Limited-memory BFGS)**:
    -   不需要显式计算和存储完整的 Hessian 矩阵。
    -   通过存储过去梯度的历史信息来近似 Hessian 矩阵的逆。
    -   在全批量和确定性模式下工作良好，但在小批量和随机性强的场景下性能下降。

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   如果损失函数在一个方向上变化很快，在另一个方向上变化很慢，梯度下降会怎么做？
-   如果损失函数有局部最小值或鞍点，梯度下降会怎么做？
-   当梯度来自小批量，存在噪声时，梯度下降会怎么做？
-   当运行 AdaGrad 很长时间时，会发生什么？
-   为什么二阶优化方法在实践中不实用？
-   在 Adam 算法中，当 $t=0$ 时会发生什么？

---
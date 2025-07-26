### [📚] 视频学习脚手架: Recurrent Neural Networks

### 一、核心内容大纲 (Core Content Outline)
-   **讲座介绍与回顾 (Lecture Introduction and Recall)**
    -   讲座主题：循环神经网络 (Recurrent Neural Networks)
    -   PyTorch 与 TensorFlow 对比回顾 (PyTorch vs TensorFlow Comparison Review)
        -   PyTorch 1.3 (Released 10/10/2019) 的主要特性 (Key Features of PyTorch 1.3)
            -   讲师个人偏好 (Personal Favorite)
            -   简洁的命令式 API (Clean, Imperative API)
            -   易于调试的动态图 (Easy Dynamic Graphs for Debugging)
            -   JIT (Just-In-Time) 允许用于生产的静态图 (JIT Allows Static Graphs for Production)
            -   支持 TPU (Tensor Processing Unit) (PyTorch/XLA) (TPU Support with pytorch/xla!) (实验性)
            -   移动端支持 (Mobile Support) (Android 和 iOS) (Experimental Mobile Support on Android and iOS!)
        -   TensorFlow 1.0 的主要特性 (Key Features of TensorFlow 1.0)
            -   默认静态图 (Static Graphs by Default) (调试可能令人困惑) (Can be confusing to debug)
            -   API 较为混乱 (API a bit messy)
        -   TensorFlow 2.0 的主要特性 (Key Features of TensorFlow 2.0)
            -   默认动态图 (Dynamic by Default)
            -   标准化 Keras API (Standardized on Keras API)
            -   发布时间较近，尚无定论 (Just came out, no consensus yet)
        -   深度学习领域变化迅速 (Field of Deep Learning Changes Rapidly)
    -   上节回顾：训练神经网络 (Last Time: Training Neural Networks)
        -   一次性设置 (One Time Setup)：激活函数 (Activation Functions)、数据预处理 (Data Preprocessing)、权重初始化 (Weight Initialization)、正则化 (Regularization)
        -   训练动力学 (Training Dynamics)：学习率调度 (Learning Rate Schedules)、超参数优化 (Hyperparameter Optimization)
        -   训练后 (After Training)：模型集成 (Model Ensembles)、迁移学习 (Transfer Learning)、大批量训练 (Large-batch Training)
-   **循环神经网络：处理序列数据 (Recurrent Neural Networks: Process Sequences)**
    -   **迄今为止的前馈神经网络 (So Far: "Feedforward" Neural Networks)**
        -   一对一 (One-to-one) 映射：例如图像分类 (Image Classification) (Image -> Label)
        -   接收单个输入，产生单个输出 (Receives a single input, produces a single output)
    -   **序列处理的范式 (Paradigms for Sequence Processing)**
        -   **一对多 (One-to-many)**：单个输入，多个输出 (Single input, sequence output)
            -   例如图像字幕 (Image Captioning)：图像 (Image) -> 单词序列 (Sequence of Words)
        -   **多对一 (Many-to-one)**：多个输入，单个输出 (Sequence input, single output)
            -   例如视频分类 (Video Classification)：图像序列 (Sequence of Images) -> 标签 (Label)
        -   **多对多 (Many-to-many)** (输入输出长度相同) (Same Length)
            -   例如逐帧视频分类 (Per-frame Video Classification)：图像序列 (Sequence of Images) -> 标签序列 (Sequence of Labels)
        -   **多对多 (Many-to-many)** (输入输出长度不同) (Different Lengths)
            -   例如机器翻译 (Machine Translation)：单词序列 (Sequence of Words) -> 单词序列 (Sequence of Words)
        -   RNN 的核心特征：能够处理任意长度的序列 (RNNs can process sequences of arbitrary length)
    -   **非序列数据的序列化处理 (Sequential Processing of Non-Sequential Data)**
        -   RNN 即使对非序列数据也很有用 (RNNs useful even for non-sequential data)
        -   **通过“瞥视”分类图像 (Classify Images by Taking a Series of "Glimpses")**
            -   RNN 顺序地查看图像的不同部分，逐步理解 (RNN looks at different parts of an image sequentially)
            -   RNN 决定查看位置是基于之前的“瞥视” (Position is conditioned on previous glimpses)
        -   **一次生成一张图像的一部分 (Generate Images One Piece at a Time)**
            -   例如 DRAW 模型 (DRAW Model) 生成图像 (Generates images)
            -   RNN 逐步“绘制”图像 (RNN paints sections of an image over time)
        -   **与油画模拟器集成 (Integrate with Oil Paint Simulator)**
            -   RNN 选择笔触来创建图像 (RNN chooses brush strokes to build up an image)
-   **循环神经网络工作原理 (How Recurrent Neural Networks Work)**
    -   **核心思想 (Key Idea)**
        -   RNN 具有“内部状态” (Internal State)，随着序列处理进行更新 (Updated as a sequence is processed)。
        -   在每个时间步 (Time Step)，RNN 接收输入 `x`，并根据旧状态 (`h_{t-1}`) 和当前输入 (`x_t`) 更新内部状态 (`h_t`)。
        -   **循环公式 (Recurrence Formula)**：`h_t = f_W(h_{t-1}, x_t)`
            -   `h_t`：新状态 (New State)
            -   `h_{t-1}`：旧状态 (Old State)
            -   `x_t`：某个时间步的输入向量 (Input Vector at some time step)
            -   `f_W`：带参数 `W` 的函数 (Some Function with parameters W)
    -   **（香草型）循环神经网络 ((Vanilla) Recurrent Neural Networks)**
        -   状态由单个“隐藏”向量 `h` 组成 (State consists of a single "hidden" vector `h`)
        -   **状态更新公式 (State Update Formula)**：`h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)` (省略偏置项) (also bias term)
            -   `W_hh`：连接前一时间步隐藏状态和当前时间步隐藏状态的权重矩阵 (Weight matrix connecting previous hidden state to current hidden state)
            -   `W_xh`：连接当前输入和当前隐藏状态的权重矩阵 (Weight matrix connecting current input to current hidden state)
        -   **输出公式 (Output Formula)**：`y_t = W_hy * h_t`
            -   `W_hy`：连接隐藏状态和输出的权重矩阵 (Weight matrix connecting hidden state to output)
        -   有时称为“香草型 RNN” (Vanilla RNN) 或“Elman RNN” (Elman RNN)，以 Jeffrey Elman 教授命名。
-   **RNN 计算图 (RNN Computational Graph)**
    -   **初始隐藏状态 (Initial Hidden State) (`h_0`)**：可设置为全零 (all 0) 或学习 (learn it)。
    -   **权重共享 (Weight Sharing)**：在序列的每个时间步重复使用相同的权重矩阵 (Re-use the same weight matrix at every time-step)。
    -   **多对多 (Many-to-Many) 任务 (例如逐帧分类) (e.g. Per-frame Classification)**
        -   在每个时间步产生一个输出 (Produce one output at each time-step)。
        -   总损失是所有时间步损失的总和 (Total loss is sum of per-time-step losses)。
    -   **多对一 (Many-to-One) 任务 (例如视频分类) (e.g. Video Classification)**
        -   只在序列末尾的最终隐藏状态产生一个输出 (Only produce one output at the final hidden state of the sequence)。
    -   **一对多 (One-to-Many) 任务 (例如图像字幕) (e.g. Image Captioning)**
        -   在第一个时间步接收单个输入 (Receive single input at the first time-step)。
        -   在后续时间步产生输出序列 (Produce output sequence at subsequent time-steps)。
    -   **序列到序列 (Sequence to Sequence) (seq2seq) 模型 (多对一 + 一对多) (Many to one + One to many)**
        -   **编码器 (Encoder)**：多对一 RNN，将输入序列编码为单个向量 (Encodes input sequence in a single vector)。
        -   **解码器 (Decoder)**：一对多 RNN，从编码后的单个向量生成输出序列 (Produces output sequence from single input vector)。
        -   编码器和解码器通常使用不同的权重矩阵 (`W_1` 和 `W_2`) (Different weight matrices for encoder and decoder).
-   **语言建模示例 (Example: Language Modeling)**
    -   **任务定义 (Task Definition)**：给定字符 `1, 2, ..., t-1`，模型预测字符 `t` (Given characters `1, 2, ..., t-1`, model predicts character `t`)。
    -   **训练序列 (Training Sequence)**: "hello"
    -   **词汇表 (Vocabulary)**: `[h, e, l, o]`
    -   **输入编码 (Input Encoding)**：将每个字符编码为独热向量 (One-Hot Vector)。
    -   **预测 (Prediction)**：在每个时间步，模型输出对词汇表中下一个字符的概率分布 (Predicts a distribution over the elements in the vocabulary for the next character)。
    -   **生成新文本 (Generate New Text)**：在测试时，从模型预测的概率分布中采样字符 (Sample characters one at a time)，并将采样的字符作为下一个时间步的输入反馈给模型 (Feed back to model)。
    -   **嵌入层 (Embedding Layer)**
        -   将独热向量 (One-Hot Vector) 乘以权重矩阵 (Weight Matrix) 相当于提取矩阵的一列 (Extracts a column from the weight matrix)。
        -   通常将此提取操作放入单独的嵌入层 (Embedding Layer) 中，以便更高效地处理稀疏的独热输入。
-   **通过时间反向传播 (Backpropagation Through Time)**
    -   **概念 (Concept)**：通过在计算图中展开整个序列来进行反向传播 (Unroll the entire sequence in the computational graph)。
    -   **步骤 (Steps)**：
        1.  正向传播 (Forward Pass) 整个序列，计算损失 (Compute loss)。
        2.  反向传播 (Backward Pass) 整个序列，计算梯度 (Compute gradient)。
    -   **问题 (Problem)**：对于长序列 (Long Sequences) 会占用大量内存 (Takes a lot of memory)。
    -   **截断通过时间反向传播 (Truncated Backpropagation Through Time)**
        -   替代近似算法 (Approximate Algorithm)。
        -   仅通过序列的“块”运行正向和反向传播 (Run forward and backward through chunks of the sequence instead of whole sequence)。
        -   将隐藏状态 (Hidden States) 永远向前传递 (Carry hidden states forward in time forever)，但只对较小步数进行反向传播 (but only backpropagate for some smaller number of steps)。
        -   在实际应用中，处理长序列时采用此方法。 (Common for long sequences).
        -   **设置 `h_0`**：处理第二个数据块时，使用第一个数据块的最终隐藏状态作为初始隐藏状态。
        -   **权重更新**：对每个块进行前向和后向传播后，立即更新权重。一旦处理完一个块，即可从内存中清除，因为所有必要信息都已存储在最终隐藏状态中。
    -   **代码实现 (Code Implementation)**：在 Python 中大约 112 行代码即可实现 (min-char-rnn.py)。
-   **莎士比亚十四行诗的例子 (The Sonnets by William Shakespeare Example)**
    -   训练一个 RNN 语言模型来处理和生成莎士比亚作品 (Train an RNN language model to process and generate Shakespeare's works)。
    -   **训练初期 (At First)**：模型生成随机字符，看起来像乱码 (Generates random garbage)。
    -   **训练更多 (Train More)**：模型开始识别结构，生成看起来像单词的序列，但仍然是无意义的句子 (Starts to recognize structure, generates word-like sequences, but still gibberish sentences)。
    -   **继续训练 (Train More)**：模型生成看起来像真实句子的文本，语法逐渐正确，甚至能学习到一些拼写错误 (Generates text that looks like real sentences, grammar improves, learns some spelling errors)。
    -   **长期训练 (Long-term Training)**：模型能够生成非常逼真的莎士比亚风格文本，包括角色对话和舞台指示 (Generates plausible Shakespearean text, including stage directions and character dialogues)。
        -   模型生成的文本听起来戏剧化，但内容仍然是无意义的。
-   **其他文本生成示例 (Other Text Generation Examples)**
    -   **代数几何教科书 LaTeX 源码 (The Stacks Project: Open-Source Algebraic Geometry Textbook LaTeX Source)**
        -   训练 RNN 生成 LaTeX 源码，模型能生成看起来像抽象数学的文本 (lemmas, proofs, diagrams)，但通常无法编译。
        -   即使手动修复编译错误，生成的数学内容也往往是无意义的。
    -   **Linux 内核 C 语言源码 (Linux Kernel C Source Code)**
        -   训练 RNN 生成 C 语言源码，模型能学习到 C 语言的结构（如函数定义、括号、缩进、`#include`、宏、常量等）。
        -   生成的代码看起来合理，包含注释，但注释内容通常无意义。
-   **搜索可解释的隐藏单元 (Searching for Interpretable Hidden Units)**
    -   研究问题：RNN 如何学习这些结构？它们的隐藏状态中捕获了什么信息？
    -   方法：选择隐藏状态的某个维度，根据其激活值（介于 -1 和 1 之间）对文本进行颜色编码。蓝色表示低激活值，红色表示高激活值。
    -   **引用 (Citation)**: Karpathy, Johnson, and Fei-Fei: Visualizing and Understanding Recurrent Networks, ICLR Workshop 2016.
    -   **例子 (Examples)**:
        -   **引用检测单元 (Quote Detection Cell)**（来自《战争与和平》）：当在引号外时，单元为蓝色；进入引号时，翻转为红色；在引号结束时，翻转回蓝色。RNN 学习了一个二进制开关来跟踪是否在引号内。
        -   **行长度跟踪单元 (Line Length Tracking Cell)**：该单元对行中的位置敏感。在回车符后重置为蓝色，并在行进过程中逐渐变红。RNN 学习了跟踪行长度。
        -   **`if` 语句单元 (`if` Statement Cell)**（来自 Linux 内核源码）：跟踪 `if` 语句内部的条件。当条件开始时，它会变蓝，并在 `if` 块内部变红。
        -   **代码深度单元 (Code Depth Cell)**：跟踪代码的缩进级别。
        -   结论：RNN 可以从数据中学习到有意义的特征和结构，即使它们只被训练来预测下一个字符。
-   **图像字幕示例 (Example: Image Captioning)**
    -   **架构 (Architecture)**：结合了卷积神经网络 (Convolutional Neural Network - CNN) 和循环神经网络 (Recurrent Neural Network)。
        -   CNN (预训练在 ImageNet 上) 用于从图像中提取特征（例如，移除最后一层）。
        -   提取的图像特征作为输入传递给 RNN。
        -   RNN 的状态更新公式中增加了图像特征的权重项：`h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + W_ih * v)`，其中 `v` 是图像特征向量。
    -   **文本生成 (Text Generation)**：
        -   从一个特殊 `<START>` 标记开始。
        -   RNN 预测第一个词，然后采样该词并作为下一个时间步的输入。
        -   重复此过程，直到模型采样到 `<END>` 标记。
    -   **结果示例 (Example Results)**：
        -   成功案例：模型能够生成非常详细和准确的图像描述（例如，“一只猫坐在地板上的手提箱里”，“一个男人骑着越野车在土路上”）。
        -   失败案例（RNN 并不那么聪明）：
            -   “一个女人手里拿着一只猫。”（实际上是穿着皮草外套的女人，可能混淆了毛茸茸的纹理）。
            -   “一个人拿着电脑鼠标在桌子上。”（实际上是拿着手机，可能由于训练集中老旧图片中经常有鼠标而产生偏见）。
            -   “一个女人站在海滩上拿着冲浪板。”（实际上是倒立的人，模型混淆了场景和动作）。
            -   “一只鸟栖息在树枝上。”（实际上是蜘蛛网在树枝上，模型混淆了对象）。
            -   “一个男人穿着棒球服扔球。”（实际上是棒球运动员俯身接球，模型在理解复杂动作上存在不足）。
        -   结论：图像字幕模型虽然令人兴奋，但仍然有局限性，特别是在理解细粒度细节和复杂场景方面。
-   **（香草型）RNN 梯度流 (Vanilla RNN Gradient Flow)**
    -   **状态更新 (State Update)**：`h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)`
    -   **反向传播 (Backpropagation)**：从 `h_t` 到 `h_{t-1}` 的梯度计算涉及乘以 `W_hh` 的转置 (actually `W_hh^T`) 和 `tanh` 的导数。
    -   **问题 (Problems for Long Sequences)**：
        1.  **梯度爆炸 (Exploding Gradients)**：如果权重矩阵 `W` 的最大奇异值大于 1，梯度在反向传播时会反复乘以一个大于 1 的值，导致指数级增长，训练不稳定（出现 `NaNs` 或 `infs`）。
            -   **控制方法 (Controlled with)**：**梯度裁剪 (Gradient Clipping)**：如果梯度的范数过大，就对其进行缩放。
        2.  **梯度消失 (Vanishing Gradients)**：如果权重矩阵 `W` 的最大奇异值小于 1，梯度在反向传播时会反复乘以一个小于 1 的值，导致指数级缩小并最终消失，模型难以学习到长距离依赖关系。

-   **长短期记忆网络 (Long Short-Term Memory - LSTM)**
    -   **发明 (Invented by)**：Hochreiter 和 Schmidhuber，1997 年。
    -   **核心思想 (Key Idea)**：每个时间步保留两个向量：**细胞状态 (Cell State)** `c_t` (记忆) 和 **隐藏状态 (Hidden State)** `h_t` (输出)。
    -   **更新规则 (Update Rule)**：更复杂，引入了“门”机制，基于 `h_{t-1}` 和 `x_t` 计算四个门值 (`i`, `f`, `o`, `g`)：
        -   `i` (输入门 - input gate): Sigmoid 激活，控制写入细胞状态的信息量。
        -   `f` (遗忘门 - forget gate): Sigmoid 激活，控制从细胞状态中擦除（遗忘）的信息量。
        -   `o` (输出门 - output gate): Sigmoid 激活，控制向隐藏状态揭示（输出）细胞状态的信息量。
        -   `g` (门控门 - gate gate): Tanh 激活，产生候选细胞状态值。
        -   **细胞状态更新 (`c_t`)**: `c_t = f ⊙ c_{t-1} + i ⊙ g` （`⊙` 表示元素级乘法）。
        -   **隐藏状态输出 (`h_t`)**: `h_t = o ⊙ tanh(c_t)`
    -   **梯度流 (Gradient Flow)**：
        -   从 `c_t` 到 `c_{t-1}` 的反向传播仅涉及元素级乘法 (`f`) 和加法，**没有矩阵乘法和非线性激活函数**（如 `tanh`）的直接路径。
        -   这形成了“梯度高速公路” (gradient superhighway)，允许梯度在时间步长上有效传播，避免梯度消失或爆炸（前提是 `f` 不太接近0或1）。
        -   这与 **残差网络 (ResNet)** 的跳跃连接思想相似，都旨在改善深层网络中的梯度流动。

-   **其他 RNN 变体 (Other RNN Variants)**
    -   **门控循环单元 (Gated Recurrent Unit - GRU)**：由 Cho 等人于 2014 年提出。
        -   LSTM 的简化版本，使用更少的门，性能与 LSTM 相似。
    -   **神经架构搜索 (Neural Architecture Search)**：
        -   例如 Jozefowicz 等人于 2015 年和 Zoph 和 Le 于 2017 年的工作。
        -   通过进化搜索 (evolutionary search) 或强化学习 (reinforcement learning) 自动发现 RNN 架构。
        -   旨在探索大量可能的更新公式，找到最优的 RNN 单元设计。
        -   虽然这些方法可以发现性能优于 LSTM 的架构，但 LSTM 和 GRU 因其性能和概念上的相对简洁性，在实践中仍被广泛使用。

### 二、关键术语定义 (Key Term Definitions)
-   **循环神经网络 (Recurrent Neural Networks - RNN)**: 一种特殊的神经网络结构，具有内部状态 (Internal State)，使其能够处理序列数据，并在处理过程中随时间更新其状态，从而学习和记忆序列的依赖关系。
-   **张量处理单元 (Tensor Processing Unit - TPU)**: 由 Google 开发的专用集成电路 (ASIC)，旨在加速机器学习工作负载，特别是神经网络的训练和推理。
-   **动态图 (Dynamic Graphs)**: 指在每次前向传播时动态构建计算图的特性，允许更大的灵活性和更容易的调试，是 PyTorch 的一个核心特点。
-   **静态图 (Static Graphs)**: 指在程序执行前预先定义好计算图的特性，编译后通常能获得更高的性能，是 TensorFlow 1.x 的主要特点。
-   **JIT (Just-In-Time) 编译 (Just-In-Time Compilation)**: 一种在程序运行时将代码编译为机器码的技术，常用于优化动态语言的性能，PyTorch 使用它将动态图转换为静态图以进行生产部署。
-   **学习率调度 (Learning Rate Schedules)**: 在神经网络训练过程中，动态调整学习率的策略，以优化模型收敛和性能。
-   **超参数优化 (Hyperparameter Optimization)**: 选择一组最优的超参数（如学习率、网络层数、隐藏单元数等）以使模型性能最大化的过程。
-   **模型集成 (Model Ensembles)**: 组合多个模型（通常是相同架构但在不同超参数或训练数据上训练的）的预测结果，以提高整体预测性能和鲁棒性。
-   **迁移学习 (Transfer Learning)**: 将在一个任务上训练好的模型（或其部分）应用到另一个相关任务上，以加速新任务的学习过程和提高性能。
-   **图像字幕 (Image Captioning)**: 一种机器学习任务，模型接收一张图像作为输入，并生成一段自然语言描述（文本序列）来描述图像内容。
-   **视频分类 (Video Classification)**: 一种机器学习任务，模型接收视频帧序列作为输入，并预测视频内容的类别或标签。
-   **机器翻译 (Machine Translation)**: 一种自然语言处理任务，模型将一种语言的文本序列翻译成另一种语言的文本序列。
-   **语言建模 (Language Modeling)**: 一种自然语言处理任务，模型学习给定历史文本序列后，预测下一个单词或字符的概率。
-   **独热向量 (One-Hot Vector)**: 一种稀疏向量表示，其中只有一个元素为1，其他所有元素为0，常用于表示分类特征，如词汇表中的单词或字符。
-   **嵌入层 (Embedding Layer)**: 神经网络中用于将离散的、高维的输入（如独热编码的词汇）映射到连续的、低维的向量空间（即词嵌入）的层。
-   **通过时间反向传播 (Backpropagation Through Time - BPTT)**: 训练循环神经网络的梯度计算算法，它通过在时间维度上展开 RNN，并对展开后的前馈网络进行标准的反向传播。
-   **截断通过时间反向传播 (Truncated Backpropagation Through Time - TBPTT)**: BPTT 的一种近似方法，为了解决长序列训练时内存和计算量过大的问题，它只在固定长度的序列块上进行反向传播，但隐藏状态会继续向前传递。
-   **梯度裁剪 (Gradient Clipping)**: 一种用于解决神经网络训练中梯度爆炸问题的技术，通过限制梯度的最大范数来防止其过大。
-   **梯度消失 (Vanishing Gradients)**: 在深度神经网络的反向传播过程中，梯度值变得极小，导致模型参数更新缓慢甚至停滞，特别是对于处理长序列的 RNN。
-   **长短期记忆网络 (Long Short-Term Memory - LSTM)**: 一种特殊的 RNN 架构，通过引入门控机制（输入门、遗忘门、输出门）和细胞状态 (Cell State) 来解决传统 RNN 的梯度消失问题，能够更好地捕捉和记忆长距离依赖。
-   **门控循环单元 (Gated Recurrent Unit - GRU)**: 另一种 RNN 架构，是 LSTM 的简化版，也使用门控机制来控制信息流，但参数量比 LSTM 少，性能相似。
-   **神经架构搜索 (Neural Architecture Search - NAS)**: 自动设计神经网络架构的过程，通常通过搜索算法（如进化算法或强化学习）来探索和评估不同的网络结构。
-   **细胞状态 (Cell State)**: LSTM 中的一个核心组件，它作为内部记忆通道，能够长期存储信息，并且信息流通过遗忘门和输入门进行控制。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **循环神经网络的运作流程 (Recurrent Neural Network Operation Flow)**:
    1.  **初始化隐藏状态 (Initialize Hidden State)**：设定初始隐藏状态 `h_0`（通常设置为全零或可学习参数）。
    2.  **序列处理 (Sequence Processing)**：在每个时间步 `t`：
        -   接收当前输入 `x_t`。
        -   结合上一时间步的隐藏状态 `h_{t-1}` 和当前输入 `x_t`，通过函数 `f_W` 更新隐藏状态为 `h_t`。
        -   根据当前隐藏状态 `h_t` 生成输出 `y_t`。
    3.  **权重共享 (Weight Sharing)**：在序列的每个时间步，都使用相同的权重矩阵 `W` 来进行计算。

-   **（香草型）循环神经网络的数学公式 ((Vanilla) Recurrent Neural Network Mathematical Formulas)**:
    -   **状态更新 (State Update)**：
        `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)` (此公式省略了偏置项，实际实现中通常包含偏置项)
    -   **输出计算 (Output Calculation)**：
        `y_t = W_hy * h_t`

-   **RNN 计算图结构 (RNN Computational Graph Structure)**:
    -   从初始隐藏状态 `h_0` 和第一个输入 `x_1` 开始，通过 `f_W` 计算 `h_1`。
    -   接着，使用 `h_1` 和 `x_2` 计算 `h_2`，以此类推，直到序列的最后一个时间步 `T`，得到 `h_T`。
    -   每个 `f_W` 节点都共享同一组权重 `W`。

-   **用于语言建模的 Python 代码示例 (Python Code Example for Language Modeling)**:
    ```python
    # 此处省略了完整的 112 行 Python 代码，但视频中指出其实现了 RNN 语言模型的训练和生成功能，不依赖 PyTorch 的自动求导。
    # 核心思想包括：
    # - 独热编码输入字符 (One-hot encoding input characters)。
    # - 手动实现循环公式 (Manually implementing the recurrence formula)。
    # - 实现通过时间反向传播 (Implementing Backpropagation Through Time)。
    # - 从输出分布中采样以生成新文本 (Sampling from output distribution to generate new text)。

    # 参考链接 (Reference Link):
    # (https://gist.github.com/karpathy/d4de566867f8291f086)
    ```

### 四、讲师提出的思考题 (Questions Posed by the Instructor)
-   对于截断通过时间反向传播 (Truncated Backpropagation Through Time)，如何设置处理第二个数据块的初始 `h_0`？
-   是否始终需要将编码器和解码器分开 (seq2seq 模型中)？

---

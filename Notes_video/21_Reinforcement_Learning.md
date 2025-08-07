### [📚] 视频学习脚手架: Lecture 21: Reinforcement Learning

### 一、核心内容大纲 (Core Content Outline)

-   **导论 (Introduction)**
    -   本节课主题：强化学习 (Reinforcement Learning)
    -   回顾之前课程内容：
        -   监督学习 (Supervised Learning)
            -   数据 (Data): (x, y)，x是数据，y是标签
            -   目标 (Goal): 学习一个函数 $f: x \to y$
            -   例子 (Examples): 分类 (Classification)、回归 (Regression)、目标检测 (Object Detection)、语义分割 (Semantic Segmentation)、图像标注 (Image Captioning)
        -   无监督学习 (Unsupervised Learning)
            -   数据 (Data): x (只有数据，没有标签)
            -   目标 (Goal): 学习数据中潜在的隐藏结构 (underlying hidden structure)
            -   例子 (Examples): 聚类 (Clustering)、降维 (Dimensionality Reduction)、特征学习 (Feature Learning) (例如：自编码器 Autoencoders)、密度估计 (Density Estimation)、生成模型 (Generative Models)
-   **什么是强化学习 (What is Reinforcement Learning)?**
    -   问题定义 (Problem Definition):
        -   一个**智能体 (agent)** 在**环境 (environment)** 中执行**动作 (actions)**，并接收**奖励 (rewards)**。
        -   目标 (Goal): 学习如何采取最大化奖励的动作 (Learn how to take actions that maximize reward)。
    -   强化学习交互循环 (Reinforcement Learning Interaction Loop):
        1.  环境 (Environment) 向 智能体 (Agent) 发送**状态 ($s_t$) (State)**。
        2.  智能体 (Agent) 基于状态选择并执行**动作 ($a_t$) (Action)**。
        3.  环境 (Environment) 向 智能体 (Agent) 发送**奖励 ($r_t$) (Reward)** (表明动作的好坏)。
        4.  动作导致环境变化，智能体接收奖励并学习 (Action causes environment change, Agent learns)。
        5.  过程重复进行 (Process repeats)。
-   **强化学习的例子 (Examples of Reinforcement Learning)**
    -   **小车-杆子问题 (Cart-Pole Problem)**
        -   目标 (Objective): 平衡移动小车顶部的杆子。
        -   状态 (State): 杆子角度 (angle)、角速度 (angular speed)、小车位置 (position)、水平速度 (horizontal velocity)。
        -   动作 (Action): 对小车施加水平力 (horizontal force)。
        -   奖励 (Reward): 每一步杆子保持直立得到 1 分。
    -   **机器人运动 (Robot Locomotion)**
        -   目标 (Objective): 使机器人向前移动。
        -   状态 (State): 所有关节的角度 (angle)、位置 (position)、速度 (velocity)。
        -   动作 (Action): 施加在关节上的扭矩 (Torques applied on joints)。
        -   奖励 (Reward): 每一步保持直立和向前移动得到 1 分。
    -   **Atari 游戏 (Atari Games)**
        -   目标 (Objective): 以最高分完成游戏。
        -   状态 (State): 游戏画面的原始像素输入 (raw pixel inputs of the game screen)。
        -   动作 (Action): 游戏控制 (Game controls) (例如：左 Left、右 Right、上 Up、下 Down)。
        -   奖励 (Reward): 每一步得分的增加/减少 (Score increase/decrease)。
        -   注意 (Note): 状态可能是不完全信息 (noisy or incomplete information)。
    -   **围棋 (Go)**
        -   目标 (Objective): 赢得比赛。
        -   状态 (State): 所有棋子的位置 (Position of all pieces)。
        -   动作 (Action): 放置下一颗棋子的位置 (Where to put the next piece down)。
        -   奖励 (Reward): 在最后一回合：如果赢了得 1 分，如果输了得 0 分 (稀疏奖励 Sparse Reward)。
-   **强化学习与监督学习的对比 (Reinforcement Learning vs. Supervised Learning)**
    -   **共同点 (Similarities)**: 都可以用迭代更新模型。
    -   **主要区别 (Key Differences)**:
        1.  **随机性 (Stochasticity)**:
            -   RL中，奖励和状态转移可能是随机的 (Rewards and state transitions may be random)。
            -   监督学习中，损失函数通常是确定性的 (Loss is typically deterministic)。
        2.  **信用分配 (Credit Assignment)**:
            -   RL中，奖励可能不是即时反馈，一个动作的奖励可能在很久之后才体现 (Reward may not directly depend on action, long-term dependency)。
            -   监督学习中，反馈是即时的 (Immediate feedback)。
        3.  **不可微 (Nondifferentiable)**:
            -   RL中，无法通过环境进行反向传播 (Cannot backpropagate through the world)。
            -   监督学习通常通过损失函数计算梯度 (Compute gradients from loss function)。
        4.  **非平稳性 (Nonstationary)**:
            -   智能体所经历的状态取决于其自身的行为，数据分布会随着智能体学习和策略变化而变化 (What the agent experiences depends on how it acts)。
            -   监督学习通常假设数据分布是静态的 (Static data distribution)。
-   **马尔可夫决策过程 (Markov Decision Process - MDP)**
    -   强化学习问题的数学形式化 (Mathematical formalization of the RL problem)。
    -   **五元组 (A tuple)**: (S, A, R, P, $\gamma$)
        -   **S (Set of possible states)**: 状态空间。
        -   **A (Set of possible actions)**: 动作空间。
        -   **R (Distribution of reward given (state, action) pair)**: 奖励函数。
        -   **P (Transition probability: distribution over next state given (state, action))**: 状态转移函数。
        -   **$\gamma$ (Discount factor)**: 折扣因子 (tradeoff between future and present rewards)。
    -   **马尔可夫性质 (Markov Property)**: 当前状态完全刻画了世界的状态。奖励和下一个状态只取决于当前状态和当前动作，而与之前的历史无关。
-   **代理的策略 (Agent's Policy)**
    -   智能体执行一个**策略 $\pi$ (policy $\pi$)**，它给出在给定状态下动作的分布。
    -   **简单 MDP: 网格世界 (Grid World)** 示例:
        -   **糟糕的策略 (Bad policy)**: 智能体在每个状态都随机向上或向下移动 (50/50 概率)，效率低下，难以达到目标状态。
        -   **最优策略 (Optimal Policy)**: 智能体在每个状态都选择最能引导其达到目标状态的动作，以最大化预期奖励。
    -   目标 (Goal): 找到最优策略 $\pi^*$ (optimal policy $\pi^*$)，使累计折扣奖励最大化 (maximizes (discounted) sum of rewards)。
        $\pi^* = \arg \max_{\pi} E \left[ \sum_{t \ge 0} \gamma^t r_t \right]$
        -   随机性来源 (Sources of randomness): 初始状态 ($s_0 \sim p(s_0)$)、动作选择 ($a_t \sim \pi(a | s_t)$)、状态转移 ($s_{t+1} \sim P(s | s_t, a_t)$) 和奖励 ($r_t \sim R(r | s_t, a_t)$)。
-   **价值函数 (Value Function) 与 Q 函数 (Q Function)**
    -   遵循策略 $\pi$ 会产生样本轨迹 (或路径) (sample trajectories or paths)：$s_0, a_0, r_0, s_1, a_1, r_1, \dots$
    -   **状态价值函数 (Value function) $V^{\pi}(s)$**:
        -   衡量一个状态有多好 (How good is a state)?
        -   定义 (Definition): $V^{\pi}(s) = E \left[ \sum_{t \ge 0} \gamma^t r_t | s_0 = s, \pi \right]$
        -   表示从状态 $s$ 开始并遵循策略 $\pi$ 所期望的累计奖励。
    -   **状态-动作价值函数 (Q function) $Q^{\pi}(s, a)$**:
        -   衡量一个状态-动作对有多好 (How good is a state-action pair)?
        -   定义 (Definition): $Q^{\pi}(s, a) = E \left[ \sum_{t \ge 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi \right]$
        -   表示在状态 $s$ 采取动作 $a$ 后，再遵循策略 $\pi$ 所期望的累计奖励。Q 函数比价值函数在数学上更方便用于学习算法。
-   **贝尔曼方程 (Bellman Equation)**
    -   **最优 Q 函数 (Optimal Q-function) $Q^*(s, a)$**:
        -   是优化策略 $\pi^*$ 的 Q 函数。它表示在状态 $s$ 采取动作 $a$ 后能获得的最大可能未来奖励 (It gives the max possible future reward when taking action a in state s)。
        -   $Q^*(s, a) = \max_{\pi} E \left[ \sum_{t \ge 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi \right]$
        -   **$Q^*$ 编码最优策略 (encodes the optimal policy)**: $\pi^*(s) = \arg \max_{a'} Q(s, a')$
    -   **贝尔曼方程 (Bellman Equation) - 递归关系 (recurrence relation)**:
        -   $Q^*(s, a) = E_{r,s'} [r + \gamma \max_{a'} Q^*(s', a')]$
            -   其中 $r \sim R(s, a)$, $s' \sim P(s, a)$
        -   **直观理解 (Intuition)**: 在状态 $s$ 采取动作 $a$ 后，我们获得奖励 $r$ 并移动到新状态 $s'$。之后，我们能获得的最大可能奖励是 $\max_{a'} Q^*(s', a')$。
-   **求解最优策略：价值迭代 (Solving for the Optimal Policy: Value Iteration)**
    -   思想 (Idea): 如果我们找到一个函数 $Q(s, a)$ 满足贝尔曼方程，那么它一定是 $Q^*$。
    -   开始时使用随机的 $Q$ 函数，并使用贝尔曼方程作为更新规则 (Start with a random $Q$, and use the Bellman Equation as an update rule):
        -   $Q_{i+1}(s, a) = E_{r,s'} [r + \gamma \max_{a'} Q_i(s', a')]$
            -   其中 $r \sim R(s, a)$, $s' \sim P(s, a)$
    -   惊人的事实 (Amazing fact): 当 $i \to \infty$ 时，$Q_i$ 收敛于 $Q^*$。
    -   问题 (Problem): 需要跟踪所有 (状态, 动作) 对的 $Q(s, a)$ 值，如果状态空间无限则不可能 (Need to keep track of $Q(s, a)$ for all (state, action) pairs – impossible if infinite)。
-   **求解最优策略：深度 Q 学习 (Solving for the Optimal Policy: Deep Q-Learning)**
    -   训练一个神经网络 (带有权重 $\theta$) 来近似 $Q^*$ (Train a neural network (with weights $\theta$) to approximate $Q^*$): $Q^*(s, a) \approx Q(s, a; \theta)$。
    -   使用贝尔曼方程来定义训练 $Q$ 的损失函数 (Use Bellman Equation to define loss function for training $Q$):
        -   $y_{s,a,\theta} = E_{r,s'} [r + \gamma \max_{a'} Q(s', a'; \theta')]$ (目标值，由目标网络 $\theta'$ 计算)
        -   损失函数 (Loss function): $L(s, a) = (Q(s, a; \theta) - y_{s,a,\theta})^2$
    -   问题 (Problem): 非平稳性 (Nonstationary)! $Q(s, a)$ 的“目标”取决于当前的权重 $\theta$ (The "target" for $Q(s, a)$ depends on the current weights $\theta$!)。
    -   问题 (Problem): 如何采样批数据进行训练？(How to sample batches of data for training?)
    -   **案例研究：玩 Atari 游戏 (Case Study: Playing Atari Games)**
        -   网络输入 (Network input): 状态 $s_t$: 最后 4 帧的 $4 \times 84 \times 84$ 堆叠图像 (经过 RGB->灰度转换、下采样和裁剪后)。
        -   网络输出 (Network output): 所有动作的 Q 值 (Q-values for all actions)。
        -   网络架构 (Network architecture): Conv(4->16, 8x8, stride 4) -> Conv(16->32, 4x4, stride 2) -> FC-256 -> FC-A (Q-values)。
        -   结果 (Results): 经过训练，算法能像专家一样玩 Atari Breakout 游戏，甚至能发现人类玩家未曾发现的策略。
-   **Q-Learning vs 策略梯度 (Policy Gradients)**
    -   Q-Learning: 训练网络 $Q_{\theta}(s, a)$ 来估计每个 (状态, 动作) 对的未来奖励。
        -   问题 (Problem): 对于某些问题，这可能是一个难以学习的函数。
    -   策略梯度 (Policy Gradients): 训练一个网络 $\pi_{\theta}(a | s)$，它将状态作为输入，并给出在该状态下采取哪个动作的分布。
    -   目标函数 (Objective function): 遵循策略 $\pi_{\theta}$ 时的期望未来奖励 (Expected future rewards when following policy $\pi_{\theta}$):
        -   $J(\theta) = E_{x \sim p_{\theta}}[ \sum_{t \ge 0} \gamma^t r_t ]$
    -   通过最大化找到最优策略 (Find the optimal policy by maximizing): $\theta^* = \arg \max_{\theta} J(\theta)$。
        -   使用梯度上升 (Use gradient ascent!)。
    -   问题 (Problem): 不可微性 (Nondifferentiability)! 不知道如何计算 $\frac{\partial J}{\partial \theta}$。
    -   **REINFORCE 算法 (REINFORCE Algorithm)**:
        -   **数学推导 (Mathematical Derivation)**:
            -   $\frac{\partial J}{\partial \theta} = E_{x \sim p_{\theta}} [ (\sum_{t \ge 0} \gamma^t r_t) \sum_{t \ge 0} \frac{\partial}{\partial \theta} \log \pi_{\theta}(a_t | s_t) ]$
            -   这个公式将梯度计算转换为期望，可以通过采样轨迹来近似。
        -   **算法步骤 (Algorithm Steps)**:
            1.  随机初始化权重 $\theta$ (Initialize random weights $\theta$)。
            2.  通过运行策略 $\pi_{\theta}$ 在环境中收集轨迹 $x$ 和奖励 $f(x)$ (Collect trajectories $x$ and rewards $f(x)$ using policy $\pi_{\theta}$)。
            3.  计算 $\frac{\partial J}{\partial \theta}$ (Compute $\frac{\partial J}{\partial \theta}$)。
            4.  对 $\theta$ 执行梯度上升步 (Gradient ascent step on $\theta$)。
            5.  跳转到步骤 2 (GOTO 2)。
        -   **直观理解 (Intuition)**:
            -   当 $f(x)$ 很高 (高奖励) 时：增加我们采取的动作的概率 (Increase the probability of the actions we took)。
            -   当 $f(x)$ 很低 (低奖励) 时：降低我们采取的动作的概率 (Decrease the probability of the actions we took)。
        -   **问题 (Problem)**: 梯度估计器方差大 (High variance of gradient estimator)。
        -   **改进策略梯度 (Improving Policy Gradients)**: 添加基线 (Add baseline) 以减少梯度估计器的方差。
-   **其他方法：基于模型强化学习 (Other approaches: Model Based RL)**
    -   **Actor-Critic (演员-评论家)**: 训练一个预测动作的演员 (actor) (如策略梯度) 和一个预测从这些动作中获得的未来奖励的评论家 (critic) (如 Q-Learning)。
    -   **基于模型 (Model-Based)**: 学习世界的状态转移函数 $P(s_{t+1} | s_t, a_t)$ 的模型，然后通过模型进行规划以做出决策。
    -   **模仿学习 (Imitation Learning)**: 收集专家在环境中执行操作的数据；学习一个函数来模仿他们的行为 (监督学习方法)。
    -   **逆强化学习 (Inverse Reinforcement Learning)**: 收集专家在环境中执行操作的数据；学习一个他们似乎正在优化的奖励函数，然后利用该奖励函数进行强化学习。
    -   **对抗性学习 (Adversarial Learning)**: 学习一个鉴别器来判断动作是真实的还是假的 (通常用于生成对抗网络)。
-   **案例研究：玩游戏 (Case Study: Playing Games)** (最新进展)
    -   **AlphaGo (2016 年 1 月)**:
        -   使用模仿学习 (Imitation Learning) + 树搜索 (Tree Search) + 强化学习 (RL)。
        -   击败了 18 届围棋世界冠军李世石 (Lee Sedol)。
    -   **AlphaGo Zero (2017 年 10 月)**:
        -   AlphaGo 的简化版本。
        -   不再使用模仿学习 (No longer using imitation learning)。
        -   击败了当时排名第一的柯洁 (Ke Jie)。
    -   **Alpha Zero (2018 年 12 月)**:
        -   推广到其他游戏：国际象棋 (Chess) 和将棋 (Shogi)。
    -   **MuZero (2019 年 11 月)**:
        -   通过游戏的学习模型进行规划 (Plans through a learned model of the game)。
        -   李世石于 2019 年 11 月宣布退役，他表示：“随着围棋 AI 的出现，我意识到即使我通过不懈努力成为第一，我也不是巅峰。即使我成为第一，也有一个无法被击败的实体。”
    -   **更复杂的游戏 (More Complex Games)**:
        -   StarCraft II: AlphaStar (2019 年 10 月): 使用多智能体强化学习达到星际争霸 II 的宗师级水平。
        -   Dota 2: OpenAI Five (2019 年 4 月): 学习玩 Dota 2 游戏。
-   **强化学习：与世界交互 (Reinforcement Learning: Interacting With World)**
    -   通常使用 RL 来训练与 (噪声、不可微分的) 环境交互的智能体 (agents)。
    -   **强化学习：随机计算图 (Reinforcement Learning: Stochastic Computation Graphs)**
        -   RL 还可以用来训练具有不可微分组件的神经网络！
        -   **示例 (Example)**: 小型“路由”网络将图像发送到 K 个网络之一。
            -   第一个 CNN 输出应该使用哪个网络的概率。
            -   从该分布中采样一个网络 (例如，绿色网络)。
            -   将图像输入到采样的网络，得到损失。
            -   将该损失作为奖励，并使用策略梯度更新第一个“路由”网络。

### 二、关键术语定义 (Key Term Definitions)

-   **强化学习 (Reinforcement Learning)**: 一种机器学习范式，其中智能体通过在环境中执行动作并接收奖励来学习如何最大化长期回报。
-   **智能体 (Agent)**: 在环境中采取行动并学习的实体。
-   **环境 (Environment)**: 智能体与之交互的系统或世界。
-   **状态 (State)**: 环境在某个时间点的描述，智能体通过它感知世界。
-   **动作 (Action)**: 智能体在环境中可以执行的操作。
-   **奖励 (Reward)**: 环境对智能体动作的即时反馈，表示该动作的好坏。
-   **策略 (Policy) ($\pi$)**: 智能体在给定状态下选择动作的规则或函数。通常表示为 $\pi(a|s)$，即在状态 $s$ 下采取动作 $a$ 的概率。
-   **折扣因子 (Discount Factor) ($\gamma$)**: 一个介于 0 和 1 之间的值，用于权衡未来奖励的重要性。
-   **马尔可夫决策过程 (Markov Decision Process - MDP)**: 强化学习问题的数学形式化，由状态、动作、奖励、状态转移函数和折扣因子定义。
-   **马尔可夫性质 (Markov Property)**: 系统的未来状态仅取决于当前状态和当前动作，而与之前的历史无关。
-   **随机性 (Stochasticity)**: 环境的特性，指奖励和状态转移可能包含随机性。
-   **信用分配 (Credit Assignment)**: 在强化学习中，将最终的奖励归因于导致该奖励的一系列动作和状态的过程。
-   **不可微 (Nondifferentiable)**: 指环境的动态或奖励函数可能无法通过传统方法进行求导。
-   **非平稳性 (Nonstationary)**: 指在强化学习中，智能体学习的数据分布会随着智能体策略的变化而变化。
-   **价值函数 (Value Function) ($V^{\pi}(s)$)**: 衡量从某个状态 $s$ 开始并遵循策略 $\pi$ 所能获得的期望累计奖励。
-   **Q 函数 (Q Function) ($Q^{\pi}(s, a)$)**: 衡量在某个状态 $s$ 采取特定动作 $a$ 后，再遵循策略 $\pi$ 所能获得的期望累计奖励。
-   **贝尔曼方程 (Bellman Equation)**: 强化学习中的一组核心方程，用于描述价值函数或 Q 函数与其未来值之间的关系，是动态规划和强化学习算法的基础。
-   **最优 Q 函数 (Optimal Q-function) ($Q^*(s, a)$)**: 代表通过采取最优行动所能达到的最大期望奖励。
-   **价值迭代 (Value Iteration)**: 一种求解最优策略的算法，通过迭代更新 Q 函数直到收敛到最优 Q 函数。
-   **深度 Q 学习 (Deep Q-Learning - DQN)**: 结合深度神经网络和 Q-Learning 的方法，使用神经网络近似 Q 函数。
-   **策略梯度 (Policy Gradients)**: 一类强化学习算法，直接优化策略函数，通常通过梯度上升来最大化期望回报。
-   **REINFORCE 算法 (REINFORCE Algorithm)**: 一种基于策略梯度的蒙特卡洛算法，通过采样轨迹来估计策略梯度。
-   **基线 (Baseline)**: 在策略梯度方法中，用于减少梯度估计器方差的参考值。
-   **Actor-Critic (演员-评论家)**: 一类强化学习算法，结合了策略梯度 (actor) 和 Q-Learning (critic) 的思想。
-   **基于模型强化学习 (Model-Based Reinforcement Learning)**: 学习环境的动态模型，然后利用该模型进行规划来做出决策。
-   **模仿学习 (Imitation Learning)**: 从专家演示中学习策略，通常采用监督学习的方式。
-   **逆强化学习 (Inverse Reinforcement Learning)**: 从专家行为中推断其潜在的奖励函数。
-   **随机计算图 (Stochastic Computation Graphs)**: 允许在神经网络中包含随机的、通常是不可微分的组件，并使用强化学习技术进行训练。
-   **硬注意力 (Hard Attention)**: 在注意力机制中，模型明确地从一组离散的位置中选择一个进行关注，这通常是不可微分的操作，需要强化学习方法进行训练。

### 三、核心算法与代码片段 (Core Algorithms & Code Snippets)

-   **最优策略目标函数 (Objective Function for Optimal Policy)**:
    $\pi^* = \arg \max_{\pi} E \left[ \sum_{t \ge 0} \gamma^t r_t \right]$

-   **价值函数 (Value Function)**:
    $V^{\pi}(s) = E \left[ \sum_{t \ge 0} \gamma^t r_t | s_0 = s, \pi \right]$

-   **Q 函数 (Q Function)**:
    $Q^{\pi}(s, a) = E \left[ \sum_{t \ge 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi \right]$

-   **最优 Q 函数的贝尔曼方程 (Bellman Equation for Optimal Q-function)**:
    $Q^*(s, a) = E_{r,s'} [r + \gamma \max_{a'} Q^*(s', a')]$
    其中 $r \sim R(s, a)$, $s' \sim P(s, a)$

-   **价值迭代更新规则 (Value Iteration Update Rule)**:
    $Q_{i+1}(s, a) = E_{r,s'} [r + \gamma \max_{a'} Q_i(s', a')]$
    其中 $r \sim R(s, a)$, $s' \sim P(s, a)$

-   **深度 Q 学习损失函数 (Deep Q-Learning Loss Function)**:
    $L(s, a) = (Q(s, a; \theta) - y_{s,a,\theta})^2$
    其中 $y_{s,a,\theta} = E_{r,s'} [r + \gamma \max_{a'} Q(s', a'; \theta')]$
    $r \sim R(s, a)$, $s' \sim P(s, a)$

-   **策略梯度目标函数 (Policy Gradients Objective Function)**:
    $J(\theta) = E_{x \sim p_{\theta}}[ \sum_{t \ge 0} \gamma^t r_t ]$

-   **策略梯度计算公式 (Policy Gradient Computation Formula)**:
    $\frac{\partial J}{\partial \theta} = E_{x \sim p_{\theta}} [ (\sum_{t \ge 0} \gamma^t r_t) \sum_{t \ge 0} \frac{\partial}{\partial \theta} \log \pi_{\theta}(a_t | s_t) ]$
    其中 $\frac{\partial}{\partial \theta} \log p_{\theta}(x) = \sum_{t \ge 0} \frac{\partial}{\partial \theta} \log \pi_{\theta}(a_t | s_t)$

-   **REINFORCE 算法 (REINFORCE Algorithm)**:
    1.  随机初始化权重 $\theta$ (Initialize random weights $\theta$)。
    2.  通过运行策略 $\pi_{\theta}$ 在环境中收集轨迹 $x$ 和奖励 $f(x)$ (Collect trajectories $x$ and rewards $f(x)$ using policy $\pi_{\theta}$)。
    3.  计算 $\frac{\partial J}{\partial \theta}$。
    4.  对 $\theta$ 执行梯度上升步 (Gradient ascent step on $\theta$)。
    5.  跳转到步骤 2 (GOTO 2)。

-   **Atari 游戏深度 Q 学习网络架构 (Atari Games Deep Q-Learning Network Architecture)**:
    -   网络输入 (Network input): 状态 $s_t$: $4 \times 84 \times 84$ 堆叠的最后 4 帧图像 (经过 RGB->灰度转换、下采样和裁剪后)。
    -   网络输出 (Network output): 所有动作的 Q 值 (Q-values for all actions)。
    -   层 (Layers):
        -   Conv(4->16, 8x8, stride 4)
        -   Conv(16->32, 4x4, stride 2)
        -   FC-256
        -   FC-A (输出 Q 值，A 为动作数量)

### 四、讲师提出的思考题 (Questions Posed by the Instructor)

-   为什么强化学习与监督学习不同？(Why is RL different from normal supervised learning?)
-   (在解释 Q 函数和价值函数后) 到目前为止是否清楚了？有任何关于 Q 函数、价值函数或任何其他内容的疑问吗？(Are we maybe clear up to this point? Any questions on these Q functions, these value functions, any any of the stuff up to this point?)
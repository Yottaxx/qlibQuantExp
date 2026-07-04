# Softmax Attention 梯度推导与分析（零跳跃极致拆解版）

本教程专为消除数学推导中的“步骤跳跃感”而设计。我们将每一个求导公式、向量乘法、求和号拆解都拆细到最基础的代数步骤，确保没有任何理解门槛。

---

## 1. 前向传播基础（符号与维度）

假设我们在处理一个包含 $B$ 只股票（或序列长度为 $B$）的横截面：
* 查询向量 (Query)：$q_i \in \mathbb{R}^d$（第 $i$ 只股票的查询特征）
* 键向量 (Key)：$k_j \in \mathbb{R}^d$（第 $j$ 只股票的键特征）
* 值向量 (Value)：$v_j \in \mathbb{R}^{d_v}$（第 $j$ 只股票的值特征）
* 特征维度：$d$，值维度：$d_v$。

### (1) 注意力 Logits $z_{ij}$
计算查询 $i$ 对键 $j$ 的缩放点积（得到一个标量）：
$$z_{ij} = \frac{q_i^T k_j}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{c=1}^d q_{i,c} k_{j,c}$$
这里 $c$ 表示向量的通道索引（从 $1$ 到 $d$）。

### (2) 注意力权重 $a_{ij}$
对第 $i$ 行的所有得分在横截面（所有的键 $j$）上做 Softmax 归一化：
$$a_{ij} = \text{softmax}_j(z_{i \cdot}) = \frac{\exp(z_{ij})}{\sum_{m=1}^B \exp(z_{im})}$$
由于归一化性质，第 $i$ 行的所有注意力权重之和恒等于 1：
$$\sum_{j=1}^B a_{ij} = 1$$

### (3) 注意力输出向量 $o_i \in \mathbb{R}^{d_v}$
将注意力权重作为系数对 Value 向量做加权求和：
$$o_i = \sum_{j=1}^B a_{ij} v_j$$
写成标量分量形式（假设 $o_i$ 的第 $c$ 个分量为 $o_{i,c}$）：
$$o_{i,c} = \sum_{j=1}^B a_{ij} v_{j,c} \quad (\text{其中 } c = 1, 2, \dots, d_v)$$

---

## 2. 损失对得分的梯度 $\frac{\partial L}{\partial z_{ij}}$ 的极细致拆解

假定 $L$ 是最终的标量损失。我们想要知道 $z_{ij}$ 的微小变化如何影响 $L$。
因为 $z_{ij}$ 影响了第 $i$ 行所有的权重 $a_{i,1}, a_{i,2}, \dots, a_{i,B}$，进而影响了输出 $o_i$，最终影响了 $L$。
所以根据**多元复合函数的链式法则**：
$$\frac{\partial L}{\partial z_{ij}} = \sum_{k=1}^B \frac{\partial L}{\partial a_{ik}} \frac{\partial a_{ik}}{\partial z_{ij}}$$
下面我们把这求和里的每一项彻底拆开计算。

---

### 步骤 2.1：计算第一部分 $\frac{\partial L}{\partial a_{ik}}$
这是损失对注意力权重的导数。
因为 $L$ 是通过输出向量 $o_i$ 的每个分量 $o_{i,c}$ 来受 $a_{ik}$ 影响的，再次使用链式法则：
$$\frac{\partial L}{\partial a_{ik}} = \sum_{c=1}^{d_v} \frac{\partial L}{\partial o_{i,c}} \frac{\partial o_{i,c}}{\partial a_{ik}}$$

我们已知输出分量公式为：
$$o_{i,c} = a_{i,1} v_{1,c} + \dots + a_{i,k} v_{k,c} + \dots + a_{i,B} v_{B,c}$$
当对特定的 $a_{ik}$ 求偏导时，只有包含 $a_{ik}$ 的那一项有导数（导数为对应的 $v_{k,c}$），其余项全部视为常数（导数为 0）：
$$\frac{\partial o_{i,c}}{\partial a_{ik}} = v_{k,c}$$

把这个结果代回：
$$\frac{\partial L}{\partial a_{ik}} = \sum_{c=1}^{d_v} \frac{\partial L}{\partial o_{i,c}} v_{k,c}$$

如果我们把损失对输出 $o_i$ 的梯度向量记为 $\gamma_i \in \mathbb{R}^{d_v}$：
$$\gamma_i = \frac{\partial L}{\partial o_i} = \left[ \frac{\partial L}{\partial o_{i,1}}, \frac{\partial L}{\partial o_{i,2}}, \dots, \frac{\partial L}{\partial o_{i,d_v}} \right]^T$$
那么上面的求和 $\sum_{c=1}^{d_v} \frac{\partial L}{\partial o_{i,c}} v_{k,c}$ 刚好就是向量 $\gamma_i$ 与 $v_k$ 的点积！
$$\frac{\partial L}{\partial a_{ik}} = \gamma_i^T v_k$$
为了后面书写简单，我们把这个标量点积记为 $g_{ik}$，它物理上代表了**值 $v_k$ 对于查询 $i$ 的“有用性”**：
$$g_{ik} \equiv \gamma_i^T v_k$$
**结论 2.1**：
$$\frac{\partial L}{\partial a_{ik}} = g_{ik}$$

---

### 步骤 2.2：计算第二部分 $\frac{\partial a_{ik}}{\partial z_{ij}}$ (Softmax 偏导)
已知：
$$a_{ik} = \frac{\exp(z_{ik})}{\sum_{m=1}^B \exp(z_{im})}$$
记分母为 $S = \sum_{m=1}^B \exp(z_{im})$。则 $a_{ik} = \frac{\exp(z_{ik})}{S}$。
我们要计算它对 $z_{ij}$ 的偏导数。根据商的求导法则 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$：
$$\frac{\partial a_{ik}}{\partial z_{ij}} = \frac{\left( \frac{\partial \exp(z_{ik})}{\partial z_{ij}} \right) \cdot S - \exp(z_{ik}) \cdot \left( \frac{\partial S}{\partial z_{ij}} \right)}{S^2}$$

由于分母 $S = \exp(z_{i,1}) + \dots + \exp(z_{ij}) + \dots + \exp(z_{i,B})$ 必定包含 $\exp(z_{ij})$ 项，所以：
$$\frac{\partial S}{\partial z_{ij}} = \exp(z_{ij})$$

现在我们必须分两种情况来看分子：

#### 情况 2.2.1：当 $k = j$ 时（对角线元素）
此时分子中的项为 $\exp(z_{ij})$。它对 $z_{ij}$ 的偏导数是其自身 $\exp(z_{ij})$。
代入商的求导公式：
$$\frac{\partial a_{ij}}{\partial z_{ij}} = \frac{\exp(z_{ij}) \cdot S - \exp(z_{ij}) \cdot \exp(z_{ij})}{S^2}$$
将分式拆开成两项相减：
$$\frac{\partial a_{ij}}{\partial z_{ij}} = \frac{\exp(z_{ij}) \cdot S}{S^2} - \frac{\exp(z_{ij}) \cdot \exp(z_{ij})}{S^2}$$
$$\frac{\partial a_{ij}}{\partial z_{ij}} = \frac{\exp(z_{ij})}{S} - \left( \frac{\exp(z_{ij})}{S} \right)^2$$
因为 $\frac{\exp(z_{ij})}{S}$ 本身就是 $a_{ij}$ 的定义，所以：
$$\frac{\partial a_{ij}}{\partial z_{ij}} = a_{ij} - a_{ij}^2 = a_{ij}(1 - a_{ij})$$

#### 情况 2.2.2：当 $k \neq j$ 时（非对角线元素）
此时分子中的项为 $\exp(z_{ik})$。因为 $k \neq j$，该项中不包含变量 $z_{ij}$，因此对 $z_{ij}$ 的偏导数直接为 0。
代入商的求导公式：
$$\frac{\partial a_{ik}}{\partial z_{ij}} = \frac{0 \cdot S - \exp(z_{ik}) \cdot \exp(z_{ij})}{S^2}$$
$$\frac{\partial a_{ik}}{\partial z_{ij}} = - \frac{\exp(z_{ik}) \exp(z_{ij})}{S^2}$$
写成两分式相乘的形式：
$$\frac{\partial a_{ik}}{\partial z_{ij}} = - \left( \frac{\exp(z_{ik})}{S} \right) \cdot \left( \frac{\exp(z_{ij})}{S} \right) = - a_{ik} a_{ij}$$

---

### 步骤 2.3：将两部分代回链式法则求和式中
我们把步骤 2.1 和步骤 2.2 的结果代入 $\frac{\partial L}{\partial z_{ij}} = \sum_{k=1}^B \frac{\partial L}{\partial a_{ik}} \frac{\partial a_{ik}}{\partial z_{ij}}$。
为了看得更清楚，我们将求和号中 $k=j$ 的项，和 $k \neq j$ 的项分离开来书写：
$$\frac{\partial L}{\partial z_{ij}} = \left( \frac{\partial L}{\partial a_{ij}} \frac{\partial a_{ij}}{\partial z_{ij}} \right) + \sum_{k \neq j}^B \left( \frac{\partial L}{\partial a_{ik}} \frac{\partial a_{ik}}{\partial z_{ij}} \right)$$

现在带入我们算好的具体导数式子：
$$\frac{\partial L}{\partial z_{ij}} = g_{ij} \cdot \left[ a_{ij}(1 - a_{ij}) \right] + \sum_{k \neq j}^B g_{ik} \cdot \left[ - a_{ik} a_{ij} \right]$$

展开第一项，并把第二项的常数 $-a_{ij}$ 提到求和符号外面：
$$\frac{\partial L}{\partial z_{ij}} = a_{ij} g_{ij} - a_{ij}^2 g_{ij} - a_{ij} \sum_{k \neq j}^B a_{ik} g_{ik}$$

现在，我们把这三项里共有的 $a_{ij}$ 提取到最外面：
$$\frac{\partial L}{\partial z_{ij}} = a_{ij} \left[ g_{ij} - a_{ij} g_{ij} - \sum_{k \neq j}^B a_{ik} g_{ik} \right]$$
对括号内的后两项提取负号：
$$\frac{\partial L}{\partial z_{ij}} = a_{ij} \left[ g_{ij} - \left( a_{ij} g_{ij} + \sum_{k \neq j}^B a_{ik} g_{ik} \right) \right]$$

注意到，括号最右侧的 $\left( a_{ij} g_{ij} + \sum_{k \neq j}^B a_{ik} g_{ik} \right)$，其实就是把原来漏掉的 $k=j$ 的项重新塞回到了求和号中。
因此，它合并后正好就是从 $1$ 到 $B$ 的完整求和：
$$a_{ij} g_{ij} + \sum_{k \neq j}^B a_{ik} g_{ik} = \sum_{k=1}^B a_{ik} g_{ik}$$
我们把这个完整的加权平均有用性记为 $\bar{g}_i = \sum_{k=1}^B a_{ik} g_{ik}$。
代回括号中，便得到了大家熟知的终极形式：
$$\frac{\partial L}{\partial z_{ij}} = a_{ij} (g_{ij} - \bar{g}_i)$$

---

### 步骤 2.4：为什么这个梯度的行和一定等于零？

这是一个非常关键的基础事实：对于固定的行 $i$，将所有列 $j$ 的梯度求和，结果必定为零：
$$\sum_{j=1}^B \frac{\partial L}{\partial z_{ij}} = \sum_{j=1}^B a_{ij} (g_{ij} - \bar{g}_i)$$
拆开求和括号：
$$\sum_{j=1}^B \frac{\partial L}{\partial z_{ij}} = \sum_{j=1}^B a_{ij}g_{ij} - \sum_{j=1}^B a_{ij}\bar{g}_i$$
在第二项中，$\bar{g}_i$ 对求和指标 $j$ 而言是常数，可以直接提出来：
$$\sum_{j=1}^B \frac{\partial L}{\partial z_{ij}} = \sum_{j=1}^B a_{ij}g_{ij} - \bar{g}_i \sum_{j=1}^B a_{ij}$$
由于注意力权重自身的性质：
1. $\sum_{j=1}^B a_{ij}g_{ij} = \bar{g}_i$ (根据平均值的定义)
2. $\sum_{j=1}^B a_{ij} = 1$ (根据权重和为 1 的定义)

代入即可得证：
$$\sum_{j=1}^B \frac{\partial L}{\partial z_{ij}} = \bar{g}_i - \bar{g}_i \cdot 1 = 0$$

---

## 3. Query 向量 $q_i$ 的梯度与“共模抵消”细节

我们想知道损失对 Query 向量 $q_i$ 的梯度。因为 $q_i$ 是一个向量：
$$q_i = [q_{i,1}, q_{i,2}, \dots, q_{i,d}]^T$$
我们需要求出它每一个分量 $q_{i,c}$ 的偏导数。根据链式法则：
$$\frac{\partial L}{\partial q_{i,c}} = \sum_{j=1}^B \frac{\partial L}{\partial z_{ij}} \frac{\partial z_{ij}}{\partial q_{i,c}}$$

### 步骤 3.1：求分量偏导数 $\frac{\partial z_{ij}}{\partial q_{i,c}}$
已知得分公式的分量形式为：
$$z_{ij} = \frac{1}{\sqrt{d}} \left( q_{i,1}k_{j,1} + \dots + q_{i,c}k_{j,c} + \dots + q_{i,d}k_{j,d} \right)$$
对特定的分量 $q_{i,c}$ 求偏导：
$$\frac{\partial z_{ij}}{\partial q_{i,c}} = \frac{k_{j,c}}{\sqrt{d}}$$

将所有分量组合回向量形式：
$$\frac{\partial z_{ij}}{\partial q_i} = \frac{k_j}{\sqrt{d}}$$

### 步骤 3.2：带入求和公式
将两部分的偏导代入：
$$\frac{\partial L}{\partial q_i} = \sum_{j=1}^B \left[ a_{ij}(g_{ij} - \bar{g}_i) \right] \frac{k_j}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) k_j$$

### 步骤 3.3：拆分 Key 并完成抵消
我们把每个股票的键 $k_j$ 分解为两部分之和：
$$k_j = k_m + kr_j$$
* $k_m$：**横截面均值（共模信号）**，所有股票共享同一个 $k_m$。
* $kr_j$：**个股相对均值的残差**。

代入公式：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) (k_m + kr_j)$$
乘法分配律展开：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \left[ \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) k_m + \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) kr_j \right]$$

因为 $k_m$ 对每个股票 $j$ 都是一模一样的，它在求和号里就相当于常数，可以直接提取到求和号外面：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \left[ k_m \left( \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) \right) + \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) kr_j \right]$$

因为我们在步骤 2.4 中严格证明过，行和项 $\sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) = 0$。
所以：
$$k_m \times 0 = 0$$
第一项完全归零消失了！
最终只剩下了个股残差项：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \sum_{j=1}^B a_{ij}(g_{ij} - \bar{g}_i) kr_j$$

---

## 4. Key 投影权重 $W_k$ 的梯度为什么对去均值免疫？

令 $k_j = W_k x_j$。我们需要计算损失对矩阵 $W_k$ 的偏导。
矩阵求导公式为：
$$\frac{\partial L}{\partial W_k} = \sum_{j=1}^B \frac{\partial L}{\partial k_j} x_j^T$$

我们首先求出对单一键向量 $k_j$ 的导数。由于 $k_j$ 在前向传播中会影响该列对应的所有得分 $z_{1,j}, z_{2,j}, \dots, z_{B,j}$：
$$\frac{\partial L}{\partial k_j} = \sum_{i=1}^B \frac{\partial L}{\partial z_{ij}} \frac{\partial z_{ij}}{\partial k_j}$$
因为 $z_{ij} = \frac{q_i^T k_j}{\sqrt{d}}$，所以 $\frac{\partial z_{ij}}{\partial k_j} = \frac{q_i}{\sqrt{d}}$。
代入得到：
$$\frac{\partial L}{\partial k_j} = \frac{1}{\sqrt{d}} \sum_{i=1}^B a_{ij} (g_{ij} - \bar{g}_i) q_i$$

### 步骤 4.1：计算所有键的梯度总和
我们将所有股票 $j$ 的键梯度相加：
$$\sum_{j=1}^B \frac{\partial L}{\partial k_j} = \sum_{j=1}^B \left( \frac{1}{\sqrt{d}} \sum_{i=1}^B a_{ij} (g_{ij} - \bar{g}_i) q_i \right)$$
交换两个求和符号的顺序（先求和 $j$，再求和 $i$）：
$$\sum_{j=1}^B \frac{\partial L}{\partial k_j} = \frac{1}{\sqrt{d}} \sum_{i=1}^B q_i \left( \sum_{j=1}^B a_{ij} (g_{ij} - \bar{g}_i) \right)$$
因为括号里的行和恒为 0，所以：
$$\sum_{j=1}^B \frac{\partial L}{\partial k_j} = 0$$

### 步骤 4.2：如果对特征进行去均值中心化
如果我们将特征去均值，令 $\tilde{x}_j = x_j - \bar{x}$（其中 $\bar{x} = \frac{1}{B} \sum_{p=1}^B x_p$ 为均值）。
此时，去均值后的权重矩阵梯度为：
$$\frac{\partial L}{\partial W_k^{\text{centered}}} = \sum_{j=1}^B \frac{\partial L}{\partial k_j} (x_j - \bar{x})^T$$
展开转置乘积：
$$\frac{\partial L}{\partial W_k^{\text{centered}}} = \sum_{j=1}^B \frac{\partial L}{\partial k_j} \left( x_j^T - \bar{x}^T \right)$$
将求和号拆开成两部分：
$$\frac{\partial L}{\partial W_k^{\text{centered}}} = \sum_{j=1}^B \frac{\partial L}{\partial k_j} x_j^T - \sum_{j=1}^B \frac{\partial L}{\partial k_j} \bar{x}^T$$
在第二项中，$\bar{x}^T$ 与求和变量 $j$ 无关，提到外面：
$$\frac{\partial L}{\partial W_k^{\text{centered}}} = \sum_{j=1}^B \frac{\partial L}{\partial k_j} x_j^T - \left( \sum_{j=1}^B \frac{\partial L}{\partial k_j} \right) \bar{x}^T$$
由于我们在步骤 4.1 中证得 $\sum_{j=1}^B \frac{\partial L}{\partial k_j} = 0$，第二项直接归零！
$$\frac{\partial L}{\partial W_k^{\text{centered}}} = \sum_{j=1}^B \frac{\partial L}{\partial k_j} x_j^T - 0 \times \bar{x}^T = \sum_{j=1}^B \frac{\partial L}{\partial k_j} x_j^T = \frac{\partial L}{\partial W_k}$$

这无可辩驳地证明了：**不论你做不做中心化去均值，权重矩阵的更新方向和大小都完全不变。**

---

## 5. 崩塌点 (Uniform Collapse) 与协方差本质的严格拆解

当注意力完全退化成等比例（均匀分布）时：
$$a_{ij} = \frac{1}{B} \quad (\text{对于所有的 } j)$$

此时，加权平均有用性变为了简单的算术平均值：
$$\bar{g}_i = \frac{1}{B} \sum_{p=1}^B g_{ip}$$

我们将这些代入 Query 梯度公式中：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \sum_{j=1}^B \frac{1}{B} (g_{ij} - \bar{g}_i) kr_j = \frac{1}{B\sqrt{d}} \sum_{j=1}^B (g_{ij} - \bar{g}_i) kr_j$$

### 步骤 5.1：引入协方差的数理定义
对于任意两个横截面序列 $X = \{x_1, \dots, x_B\}$ 和 $Y = \{y_1, \dots, y_B\}$，其协方差的定义公式为：
$$\text{Cov}(X, Y) = \frac{1}{B} \sum_{j=1}^B (x_j - \bar{X})(y_j - \bar{Y})$$

现在，我们把我们的变量与定义一一对齐：
* 令 $x_j = g_{ij}$。它的均值 $\bar{X} = \bar{g}_i = \frac{1}{B} \sum_{p=1}^B g_{ip}$。
* 令 $y_j = kr_j$。由于 $kr_j$ 本身是键特征偏离均值的残差，因此它的均值 $\bar{Y} = 0$。

代入协方差公式：
$$\text{Cov}_j(g_{ij}, kr_j) = \frac{1}{B} \sum_{j=1}^B (g_{ij} - \bar{g}_i) (kr_j - 0) = \frac{1}{B} \sum_{j=1}^B (g_{ij} - \bar{g}_i) kr_j$$

### 步骤 5.2：最终化简形式
将这个协方差关系代回到 Query 的梯度表达式中：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \cdot \left[ \frac{1}{B} \sum_{j=1}^B (g_{ij} - \bar{g}_i) kr_j \right] \times B \quad (\text{错误，应该是直接对齐})说明如下：$$
由于 $\text{Cov}_j(g_{ij}, kr_j)$ 刚好包含了 $\frac{1}{B}$ 这个系数，因此：
$$\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d}} \text{Cov}_j(g_{ij}, kr_j)$$

**这就是核心结论**：当注意力崩塌时，Query 能否获得有效的学习信号，完全取决于**个股特征的偏离程度 $kr_j$** 与**该个股信息的有用性 $g_{ij}$** 在横截面上的协方差。

* 如果没有协方差（例如有用性杂乱无章或大家都一样），梯度就为 0，Query 彻底“冷掉”，网络失去学习能力。

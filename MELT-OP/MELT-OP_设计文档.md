# MELT-OP：基于潜在堆积相变动力学的有机分子熔点预测

> 版本：v0.1（理论设计稿）  
> 输入：分子 SMILES  
> 监督：有机分子实验熔点 \(T_m\)  
> 目标：在不依赖晶体结构、自由能曲线、焓、熵或其他热力学标签的条件下，构建一种不同于传统“分子编码器 + 回归头”的熔点预测神经网络。

## 1. 研究动机

有机分子的熔点不是单纯的局部官能团属性，而是分子结构、分子间互补、对称性、刚柔性、潜在堆积方式和实验噪声共同作用的结果。

当前常见方法大体遵循：

\[
G=\operatorname{Encoder}(\mathrm{SMILES}),\qquad
\hat T_m=\operatorname{MLP}(G)
\]

这类模型即使使用 GNN、Transformer、3D 构象或多构象集成，本质上仍是静态表征后的点回归。另一个直觉方案是学习固相与液相的自由能曲线，但在只有 SMILES 和熔点标签的数据条件下，缺少自由能、焓、熵、晶体结构等监督，容易出现不可辨识的伪物理量。

MELT-OP 不尝试恢复真实自由能曲线，而是重新定义预测问题：

> 不直接学习 \(f(\mathrm{SMILES})=T_m\)，而是学习由分子结构诱导的潜在堆积状态动力学，并把温度响应的临界位置作为熔点预测。

这里的潜变量是“可用于预测的有效堆积/有序状态”，不是实验晶型，也不是自由能。

## 2. 核心范式

MELT-OP（Molecular Emergent Lattice-Transition Operator，分子涌现晶格转变算子）包含四个阶段：

\[
\mathrm{SMILES}
\rightarrow
\{\text{潜在堆积微态}\}
\rightarrow
\text{温度驱动的状态动力学}
\rightarrow
\text{隐式相变场}
\rightarrow
\hat T_m
\]

其核心计算对象不是一次性的原子 embedding，而是一个由分子结构条件化的状态转移算子。

### 2.1 SMILES 到分子状态

将 SMILES 转为带原子、键、环、形式电荷和立体化学信息的分子图 \(G=(V,E)\)。

对原子 \(i\) 生成初始特征：

\[
h_i^0 =
\operatorname{TokenEncoder}
(\text{atom}_i,\text{bond}_i,\text{ring}_i,
\text{charge}_i,\text{stereo}_i)
\]

为了保留全局拓扑关系，额外计算图拉普拉斯或随机游走基：

\[
u_i=[\phi_1(i),\phi_2(i),\ldots,\phi_d(i)]
\]

\(u_i\) 不是三维坐标，而是分子图上的全局拓扑坐标，用于表达环系、支化、长距离连接和对称性。

### 2.2 潜在堆积微态

模型生成 \(K\) 个可重参数化的潜在微态：

\[
z_k = \operatorname{StateEncoder}(G,\xi_k),
\quad k=1,\ldots,K
\]

其中 \(\xi_k\) 是随机扰动。每个 \(z_k\) 表示一种“预测上可能的有效堆积模式”，不要求与某个真实晶型一一对应。

由 \(z_k\) 产生低秩接触/交换算子：

\[
A_k=\operatorname{softmax}(P_kP_k^\top+B_k)
\]

其中 \(A_k\) 可以学习分子内部片段之间的协同关系，例如：

- 氢键和偶极互补；
- 芳香面之间的相容性；
- 刚性片段共面；
- 疏水表面匹配；
- 柔性链段的协同冻结；
- 对称性带来的重复排布。

\(A_k\) 不是由真实晶体接触图监督，而是由熔点标签、结构不变性和动力学约束端到端诱导。

### 2.3 温度驱动的潜在动力学

对每个微态和无量纲温度 \(\tau\in[0,1]\)，运行 \(T\) 步可微状态迭代：

\[
m_{k,\tau}^{t+1}
=
m_{k,\tau}^{t}
+
\eta\Psi_\theta
(m_{k,\tau}^{t},A_k,\tau,r)
\]

其中：

- \(m\)：潜在有序状态；
- \(A_k\)：微态对应的交换算子；
- \(\tau\)：归一化温度；
- \(r\)：从 SMILES 学到的分子调制向量；
- \(\Psi_\theta\)：可学习的状态转移算子。

一种可实现的转移形式为：

\[
\Psi_\theta(m,A,\tau,r)
=
g_\theta(r,\tau)\odot(Am-m)
-\alpha_\theta(r,\tau)m
+\beta_\theta(r,\tau)\tanh(m)
+\gamma_\theta(r,\tau)\lVert m\rVert^2m
\]

它不声称是分子动力学或真实热力学方程，而是学习“结构条件下的有序状态如何随温度变化”。

### 2.4 响应峰读出

定义温度响应函数：

\[
\chi_k(\tau)
=
\frac{\lVert
m_k(\tau+\delta)-m_k(\tau-\delta)
\rVert^2}{2\delta}
\]

响应峰代表模型内部状态发生快速转变的位置。用 soft-argmax 得到第 \(k\) 个微态的临界温度：

\[
T_k
=
T_{\min}
+
(T_{\max}-T_{\min})
\sum_l
\operatorname{softmax}
\left(\frac{\chi_k(\tau_l)}{s}\right)\tau_l
\]

然后按微态权重 \(\pi_k\) 聚合：

\[
\mu=\sum_k\pi_kT_k
\]

结构不确定性和观测噪声分别建模为：

\[
\sigma^2=
\underbrace{\sigma_{\text{aleatoric}}^2}_{\text{实验/测量噪声}}
+
\underbrace{\sum_k\pi_k(T_k-\mu)^2}_{\text{潜在堆积模式差异}}
\]

## 3. 更强的输出形式：隐式相变场

为了进一步避免“动力学最后接一个回归头”，可以直接学习分子条件相变场：

\[
S_\theta(G,T)\in(0,1)
\]

其含义是：给定分子 \(G\) 在温度 \(T\) 下仍处于有序固相的软占据概率。

熔点定义为：

\[
S_\theta(G,T_m)=0.5
\]

预测时通过可微二分搜索或 differentiable root finding 求解 \(\hat T_m\)。

为了保证温度升高时固相占据不反常增加，可写成：

\[
S_\theta(G,T)=\sigma[-u_\theta(G,T)]
\]

并约束：

\[
u_\theta(G,T)
=
\sum_r a_r(G)\operatorname{softplus}(\phi_r(T))+b(G),
\qquad a_r(G)\ge 0
\]

因此 \(S_\theta\) 对温度单调下降。

还可以定义条件熔点分布：

\[
p_\theta(T\mid G)
=
-\frac{\partial S_\theta(G,T)}{\partial T}
\]

这样模型输出的不再是单个温度，而是可计算均值、分位数和置信区间的温度分布。

多微态版本为：

\[
S_{\text{mix}}(G,T)
=
\sum_{k=1}^K\pi_k(G)S_k(G,T)
\]

在有重复实验值或多晶型证据时，可用模式熵表达结构不确定性；只有单标签时应使用稀疏/单峰先验，防止模型任意制造多峰。

## 4. 训练目标

总损失可以写成：

\[
\mathcal L=
\mathcal L_{\text{phase}}
+\lambda_{\text{view}}\mathcal L_{\text{view}}
+\lambda_{\text{rank}}\mathcal L_{\text{rank}}
+\lambda_{\text{local}}\mathcal L_{\text{local}}
+\lambda_{\text{div}}\mathcal L_{\text{div}}
+\lambda_{\text{peak}}\mathcal L_{\text{peak}}
+\lambda_{\text{smooth}}\mathcal L_{\text{smooth}}
\]

### 4.1 相变场监督

只有实验熔点 \(T_{m,i}\) 时，可构造由相变定义得到的软边界：

- \(T<T_{m,i}-\Delta\)：目标接近 1；
- \(T>T_{m,i}+\Delta\)：目标接近 0；
- \(T\approx T_{m,i}\)：要求 \(S\) 接近 0.5。

\(\Delta\) 仅表示测量误差和标签不确定带，不是额外热力学标签。

可以同时使用：

\[
\mathcal L_{\text{root}}
=
\left(S_\theta(G_i,T_{m,i})-0.5\right)^2
\]

以及温度采样点上的 BCE 或软标签损失。

若直接采用分布建模，则用 Student-t NLL 或 CRPS：

\[
\mathcal L_{\text{pred}}
=
-\log p_\theta(T_{m,i}\mid G_i)
\]

Student-t 比 MSE 更适合存在文献噪声和异常值的熔点数据。

### 4.2 SMILES 多视图一致性

对同一分子生成多个等价视图：

- canonical SMILES；
- 随机原子顺序；
- 等价 Kekulé 形式；
- 合法局部重写；
- 轻度可逆 token dropout。

要求输出的微态分布一致：

\[
\mathcal L_{\text{view}}
=
D_{\mathrm{KL}}
\left(q(z\mid g_1)\,\Vert\,q(z\mid g_2)\right)
\]

这可以防止模型记忆 SMILES 字符串顺序。

### 4.3 局部排序和邻域平滑

对于结构相似分子 \(i,j\)，如果 \(T_i>T_j\)，加入：

\[
\mathcal L_{\text{rank}}
=
-\log\sigma
\left(
\frac{\mu_i-\mu_j}{\tau_r}
\right)
\]

权重可以由 ECFP 相似度、实验来源和标签置信度共同决定。

### 4.4 微态多样性

使用排斥核和模式熵防止 \(z_1,\ldots,z_K\) 塌缩：

\[
\mathcal L_{\text{div}}
=
\sum_{i\ne j}
\exp\left(-\frac{\lVert z_i-z_j\rVert^2}{\rho^2}\right)
-\lambda H(\pi)
\]

单标签数据中不能强行要求每个模式产生不同熔点，而应鼓励其保留不同结构假设并通过分布不确定性表达。

### 4.5 响应峰与平滑正则

\[
\mathcal L_{\text{peak}}
=
\operatorname{width}(\chi)
+
\lambda_{\text{spurious}}
\operatorname{num\_spurious\_peaks}(\chi)
\]

\[
\mathcal L_{\text{smooth}}
=
\sum_l
\left\|
\chi(\tau_{l+1})-2\chi(\tau_l)+\chi(\tau_{l-1})
\right\|^2
\]

该约束允许临界区快速变化，但抑制温度轴上的随机振荡。

## 5. 最小可行版本：MELT-OP-Lite

建议先实现一个可控版本：

- 原子/键隐藏维度：128；
- 全局图谱基维度：16；
- 微态数：\(K=4\)；
- 动力学步数：\(T=8\)；
- 温度网格：32 个点；
- 潜在有序状态维度：64；
- 输出：Student-t NLL + 视图一致性 + 微态多样性；
- 首轮不加入外部描述符，不加入 3D 坐标；
- 使用 scaffold split 作为主测试协议。

训练顺序建议：

1. 先训练 \(K=1\)，验证隐式相变场稳定；
2. 逐步增加到 \(K=3\) 或 \(K=4\)；
3. 对 soft-argmax 温度参数退火；
4. 最后加入多模式不确定性和重复标签 likelihood。

## 6. 必须设置的基线与消融

### 6.1 参数量匹配基线

至少比较：

- ECFP + XGBoost；
- D-MPNN/Chemprop；
- GIN/GAT；
- Graphormer；
- SMILES Transformer；
- 若允许生成构象，再比较 SchNet/Uni-Mol 类模型。

所有模型使用相同数据清洗、相同主划分和 5 个随机种子。

### 6.2 核心消融

1. 完整 MELT-OP；
2. 去掉温度动力学，仅保留静态回归；
3. 将动力学轨迹替换为一次 pooling + MLP；
4. 去掉响应峰，直接用最终隐藏向量回归；
5. 去掉微态集合，仅保留 \(K=1\)；
6. 固定或随机化 \(A_k\)；
7. 去掉 SMILES 多视图一致性；
8. 去掉单调相变场；
9. MSE 与 Student-t NLL 对比。

如果“动力学 + 临界读出”被替换成普通 pooling 后性能几乎不变，则不能宣称新范式有效。

### 6.3 OOD 测试

主结果使用 scaffold split，并额外测试：

- Bemis–Murcko scaffold holdout；
- ECFP Tanimoto \(\ge 0.8\) 聚类分组；
- 官能团留一；
- 同系物留一；
- 分子量/环数区间外推；
- 外部数据集。

报告：

- MAE、RMSE、\(R^2\)；
- Spearman/Kendall；
- NLL、CRPS；
- 90% 区间覆盖率；
- ECE calibration；
- 高温/低温分层 MAE；
- scaffold bootstrap 置信区间。

## 7. 数据处理原则

1. 先按 canonical SMILES/InChIKey 去重；
2. 同一分子的重复测量全部放入同一个数据划分；
3. 盐、溶剂化物、互变异构体、立体化学缺失单独标记；
4. 记录原始实验来源和单位，统一转换为 K；
5. 对异常值不直接删除，优先使用 Student-t/Huber；
6. 训练、验证和测试不能共享近重复骨架；
7. scaffold split 作为主要结论，random split 仅作为参考。

## 8. 理论新颖性判据

要让论文中的贡献不被认为是“GNN + Transformer + 动力学”的简单拼接，需要同时满足：

1. **新的计算对象**：状态空间是分子级 latent packing states，而不是普通节点 embedding；
2. **新的内层运算**：通过状态交换核和温度驱动动力学寻找临界响应；
3. **新的监督方式**：从点回归改为受单调性约束的隐式相变场；
4. **新的推理方式**：通过求相变场根/响应峰得到熔点，而不是一次 pooling + MLP。

建议做一个表达能力单元测试：构造局部 \(k\)-hop 邻域相似但全局环/模块连接不同的分子对，比较普通有限深度 MPNN、静态 pooling 和 MELT-OP 对全局结构变化的区分能力。

## 9. 风险与边界

- \(z_k\)、\(A_k\)、\(m\) 和 \(S(G,T)\) 都是预测潜变量，不能直接称为真实晶型、自由能或实验序参量；
- 没有额外热力学标签时，潜变量存在不可辨识性，必须通过视图一致性、模式多样性和单调性限制；
- 数据规模较小时应减少 \(K\)、动力学步数和隐藏维度，避免过拟合；
- 多晶型标签若没有重复实验支持，不宜任意制造多峰输出；
- 盐、溶剂化物、互变异构体和立体化学缺失应分层报告；
- 随机 split 上的轻微提升不能证明新范式成立，重点应看 scaffold/OOD 和参数量匹配结果。

## 10. 推荐论文表述

建议将方法定义为：

> Latent Packing Phase Dynamics for Melting-Point Prediction from SMILES

中文：

> 基于潜在堆积相变动力学的有机分子熔点预测网络

方法学核心不是增加更多描述符，而是改变预测问题：

\[
\text{传统：}\quad f(\mathrm{SMILES})=T_m
\]

\[
\text{MELT-OP：}\quad
\mathrm{SMILES}
\rightarrow
\text{latent packing dynamics}
\rightarrow
S(G,T)
\rightarrow
S(G,\hat T_m)=0.5
\]

最终能否取得 SOTA，需要由严格 scaffold split、外部测试、参数量匹配和消融实验验证，不能在理论设计阶段预先承诺。

## 11. 下一步实现计划

### 阶段 A：验证相变场

- 实现 SMILES 图编码器；
- 实现单模式 \(K=1\)；
- 实现单调隐式场和可微二分求根；
- 与静态 MLP 回归比较。

### 阶段 B：加入潜在堆积微态

- 增加 \(K=3\sim4\)；
- 引入低秩交换算子 \(A_k\)；
- 加入视图一致性和模式多样性；
- 观察不确定性校准变化。

### 阶段 C：完整评测

- 参数量匹配基线；
- scaffold、cluster、官能团留一测试；
- 高低熔点分层；
- 重复测量和多晶型子集；
- 发布训练脚本、配置和消融表。

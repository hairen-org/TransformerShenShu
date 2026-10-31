# 架构说明：Shenshu-42M-v1

## 目标

内容对齐：同一 四柱 / 六亲 输入 → 模型预测的六门条文号经条文库查表后，与古典十九法命书逐条一致。

## 映射

\[
x=(\text{sex},\text{pillars},\text{qin}) \xrightarrow{\mathrm{Enc}} h
\xrightarrow{6\times\mathrm{Head}} \hat{y}_{1..6}\in\{0..n_{bars}-1\}
\xrightarrow{\mathrm{LUT}} \text{命书}
\]

## 模块

| 模块 | 默认 | 说明 |
|------|------|------|
| Input embedding | 干支 / 生肖 / 人数 类别 embedding + 槽位 embedding，d_model=256 | 15 个结构化槽位 |
| Encoder | 4 层 Transformer，nhead=4，FFN 512 | mean pool 得到 h |
| 真刻 MLP | 父·母·兄弟·配偶 embedding 拼接 → 8 类 | 对应古典「六亲考刻」 |
| 刻 embedding | 8 × d_model | 训练时 teacher forcing 真刻，推理时用预测真刻 |
| Bar trunk + 6 heads | 共享 trunk + 门 embedding，每头 12000 类 | 输出空间即条文号（class id = 条文数 − 1001） |
| 参数量 | 21,096,520 | |

## 损失

- `loss = 2.0 · CE(真刻) + mean CE(六门条号)`
- 门控指标：验证集六门条号平均准确率 ≥ 0.95（实测 0.9987）。

## 可选模块

`use_vq` / `use_lm` 可开启 VQ 瓶颈与条件 LM decoder（默认关闭），用于生成近似措辞的变体，不作为内容对齐的门控。

# 双机器人情绪感知社交导航 —— 训练改动记录与实验计划

> 记录针对双机器人（robot_num=2）情绪感知社交导航任务的所有落地改动、
> 创新点、预期实验与结果，便于后续回看与撰写论文。
> 所有分数一律用固定 seed 的 test 模式（500 episode）报告，保证可复现。

## 1. 项目背景与创新点
基于 EmoBipedNav（情绪感知社会导航，Georgia Tech）的 SAC-AE 框架，
扩展为**双移动机器人同时导航**设定。

### 1.1 创新点（论文卖点）
1. **双机器人情绪感知社会导航**：两个双足（LIP/Digit）机器人在拥挤行人中
   同时导航，决策融入行人情绪（LiDAR 几何层提取的 emotion layer），并要求
   两机协同完成导航。
2. **机器人间隐式协调**：
   - 观测加入他机相对状态特征（自车系下距离/角度/相对速度），4x(N-1) 维；
   - 奖励加入机器人间社交距离不适惩罚（discomfort penalty）；
   - 去中心化执行：每机独立运行自身策略。
3. **局部情绪感知**：情绪统计只在 LiDAR 可视范围内计算，更符合真实感知约束。
4. **CTDE 集中式训练/去中心化执行**（可选开关）：共享全局 Critic 缓解多机非平稳。
5. **测试期高层仲裁**（可选）：LLM/规则层在机机冲突时调制让行速度。

### 1.2 与单机器人基线的关系
单机器人是 sanity baseline（参考注释：SAC-AE ~90%@2.74M、LNDNL ~88%@480k、
DRL-VO ~90%@2.74M）。双机器人是扩展贡献，任务更难（两机都需到达目标），
分数预期低于单机，但应显著高于"双机朴素展开/无协调"对照。

## 2. 问题诊断（为什么当前成功率只有 0.03）
### 2.1 直接原因：续训冷启动
- Replay buffer 从未持久化（buffer 目录为空），每次续训 buffer 从零开始；
- 续训头 10k 步只采集不更新；
- 结果：每次"中断->续训->首次评估"都暴跌（400k->0.01、1.2M->0.03），历史最高仅 0.32。

### 2.2 结构性原因
- 双机独立学习 + 两机都需到达（all_reach）-> 非平稳、难度高；
- decoder_type=identity：SAC-AE 重建损失被关闭，纯 TD 训练像素编码器，不稳定；
- buffer=30k 偏小（内存约束），off-policy 多样性受限；
- 训练评估为 100 个随机 episode，噪声大。

### 2.3 已修复的代码问题
1. 多机 done 标志被统一覆盖 -> 改为按机器人独立 done；
2. 已到达机器人每步重复获得 success_reward -> 仅首次到达给奖；
3. 续训 10k 步空转 -> 恢复 buffer 后短暖机（+1000）；
4. 续训重设种子 -> 跳过，保持场景连续。

## 3. 代码改动清单
### 3.1 续训无损化（emobipednav_main.py）
- A. 中断时保存 buffer（`if _INTERRUPTED:` 块）
- B. 续训时恢复 buffer（`--load_model` 块）
- C. 缩短续训暖机（`update_start` 改为 +1000）
- D. 续训不重设种子（`set_seed_everywhere` 仅 fresh run）
- E. 每个评估点也存 buffer（防 kill -9）
### 3.2 多智能体正确性
- emobipednav_main.py：done_bool -> 每机独立 done_list
- env_emobipednav.py：`reached_list` 使到达奖励只发一次
（具体补丁见文末。）

## 4. 预期实验设计
### 4.1 主实验
双机器人 SAC-AE（robot_num=2），训练到收敛，报告 success/collision/nav time。

### 4.2 消融实验
| 实验 | 配置 | 目的 |
|------|------|------|
| 无协调 | `--disable_robot_relative` | 验证 inter-robot 特征+奖励贡献 |
| 无情绪 | `--disable_emotion` | 验证情绪感知贡献 |
| 单机基线 | `--robot_num 1` | 对齐参考 0.9，sanity check |
| CTDE | `--use_centralized_critic` | 验证集中式 critic 缓解非平稳 |

### 4.3 基线对照
DWA、DRL-VO、LIDAR-SAC、LNDNL（repo 自带 benchmark）双机设定下对照。

### 4.4 指标
- 主指标：success_rate（两机都到达 / 测试集总数），test 模式固定 seed 500 集
- 辅助：collision_rate、avg_nav_time、timeout_rate

## 5. 预期结果（占位，跑完用真实数据填充）
| 方法 | Success | Collision | Nav time | 备注 |
|------|---------|-----------|----------|------|
| 双机 SAC-AE（本文） | 待填 | 待填 | 待填 | 主结果 |
| - 无协调 | 待填 | 待填 | 待填 | 消融 |
| - 无情绪 | 待填 | 待填 | 待填 | 消融 |
| 单机 SAC-AE | ~0.9 | 待填 | 待填 | 基线 |
| DWA / DRL-VO / LNDNL | 待填 | 待填 | 待填 | 基线 |

## 6. 训练/续训/测试流程
```bash
# 训练（6h 一段，Ctrl+C 结束，下次用 step_XXX_interrupt 续）
python emobipednav_main.py --robot_num 2 --robot_model lip --robot_eval_model lip

# 续训
python emobipednav_main.py --robot_num 2 --robot_model lip --robot_eval_model lip \
    --load_model step_XXX_interrupt

# 出正式分数（固定 500 episode）
python emobipednav_main.py --robot_num 2 --robot_model lip --robot_eval_model lip \
    --load_test_model step_XXXXXX_success_XX
```

## 7. 里程碑与时间规划
- 每个 session 约 6h，lip 模型约 20 万步
- 先跑单机 baseline 确认框架正常（对齐 0.9）
- 再跑双机主实验 + 消融，预算 2-6M 步，分多个 session 续训

## 8. 回滚说明
每项改动可单独回退（见文末补丁标注）。

## 附录：代码补丁
（补丁内容由实现会话按 emobipednav_main.py / env_emobipednav.py 逐条落地：
A-F 在 emobipednav_main.py，G-I 在 env_emobipednav.py，见代码注释。）

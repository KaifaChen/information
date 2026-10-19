# information scripts —《条件独立与信息约束方程》补充材料

本目录为论文《条件独立与信息约束方程：四泛函的导出、闭合性与自然梯度动力学》的数值核验脚本与数据（随文提交）。

## 文件与论文各节的对应

| 文件 | 核验内容 | 论文位置 |
| :--- | :--- | :--- |
| `examples_check.py` | 笔算实例核验：两态机（π、h_μ、H(τ)、I、C_μ、P_b、逐转移 ΔS、同步发射 ΔS）、三态反例（ΔS(A→C)）、两系统一致性实例（A） | §3.3、§3.4、§4.3 |
| `appendix_b_jacobian.py` | 附录 B 的族级独立性核验：四泛函 (H, I, ΔS, A) 在三状态环流族 θ*=(0.3,0.1,0.7,0.5,0.3) 处的雅可比矩阵（偏导表、奇异值、行秩 4、两个 2×2 子式） | 附录 B |
| `脚本\kx_lattice.py` | 单单元两变量 Shannon 格闭合检验（T-K1 五条恒等式、T-K2 六量三参数闭合、T-K3 轴间共线表） | §5.1 |
| `结果\route_func3.tsv` | kx_lattice.py 的输入数据：代码域函数级单元表（92 个合格单元，add-1 平滑） | §5.1 |

## 运行方式

```bash
# 笔算实例核验（§3.3、§3.4、§4.3）
python examples_check.py                   # 输出同时写入 examples_check.txt

# 附录 B：雅可比与秩 4 独立性核验
python appendix_b_jacobian.py              # 输出偏导表、奇异值、行秩

# 格恒等式核验（§5.1）
python 脚本/kx_lattice.py                 # 输出同时写入 结果/kx_lattice.txt
```

依赖：除 `appendix_b_jacobian.py` 需 NumPy 外，其余脚本仅用 Python 3 标准库。

## 预期输出（examples_check.py）

```
pi = (0.4000, 0.6000)
H(tau|0) = 0.6109 ; H(tau|1) = 0.5004 ; h_mu = 0.5446
Pbar = (0.4000, 0.6000) ; H(tau) = 0.6730 ; I = 0.1284 ; C_mu = 0.6730
P_b(.|0) = (0.5500, 0.4500) ; P_b(.|1) = (0.3000, 0.7000)
P_b(.|0) vs P_f(.|0): NOT equal ; P_b(.|1) vs P_f(.|1): NOT equal
DeltaS(0->1) = 0.3389 ; DeltaS(1->0) = 0.2580 ; DeltaS(0->0) = 0.0472 ; DeltaS(1->1) = 0.0257
sync emission E=delta: P_b(.|1) = (0.2000, 0.8000) ; P_b(.|1) vs P_f(.|1): NOT equal ; DeltaS(0->1) = 1.6094 (= ln 5)
DeltaS(A->C) = 0.6931 (= ln 2)
A = 0.8014
```

## 复现约定

- `appendix_b_jacobian.py`：θ*=(0.3,0.1,0.7,0.5,0.3) 处中心差分、步长 1e-6；不含随机数，确定性输出，精确复现附录 B.1 的雅可比表。
- `kx_lattice.py`：数据 `route_func3.tsv`（add-1 平滑）；双运行逐字节一致后锁定；T-K1 判据为五条恒等式最大偏差 < 1e-12。

## 校验和（SHA256，锁定件）

```
210BFFC68740B8FB5E883959C7893159CEBF3F67BBC17630768A75DE164B39F2  examples_check.py
F769C0C4683DF2B9E75E68FF9D6B34E21F74F3BBDDC895D4B47D1BEF294DC51F  appendix_b_jacobian.py
3359BBEAC7D9F8ADD9DBA577BAB4CF98B77BCFE40DF66F1C5C40D205AC838D5D  脚本/kx_lattice.py
F8A8F4F9796C80C77AC33E9B894C59323F9CEF94F7BFC1AD2F90CACA20E007C3  结果/route_func3.tsv
```

## 冒烟测试（2026-10，通过）

- `examples_check.py`：复现论文全部笔算数字（h_μ=0.5446、I=0.1284、ΔS(0→1)=0.3389、同步发射 ΔS(0→1)=ln5、ΔS(A→C)=ln2、A=0.8014）；显式输出两态机 P_b(·|s)≠P_f(·|s)（细致平衡是状态链可逆性，不蕴含符号层同点可逆）；双运行逐字节一致；
- `appendix_b_jacobian.py`：精确复现附录 B.1 的雅可比（H 行：0, 0, −0.513482, −0.231049, +0.051384；奇异值 0.921121/0.363167/0.140093/0.069091；行秩 4；两个子式 −0.036665 与 −0.025870）；
- `kx_lattice.py`：K=92，T-K1 最大偏差 7.105e-15，T-K2 闭合，T-K3 H(p)–I ≈ +0.97；双运行逐字节一致（SHA256 807675E10BCC1E233E9D88216B9B3812E14FD3221A025FC664438C8BFBBFEE62）。

## 许可证

见 [LICENSE](LICENSE)（研究与教学用途；商业使用请联系作者）。

> AI 使用声明：本 README 使用 Deepseek AI 辅助编写；文件校验和由 Get-FileHash 生成，作者已核对。

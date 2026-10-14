# information scripts —《论信息的约束方程》补充材料

本目录为论文《论信息的约束方程：四泛函的导出、闭合性与动力学》的数值核验脚本与数据（随文提交）。

## 文件与论文各节的对应

| 文件 | 核验内容 | 论文位置 |
| :--- | :--- | :--- |
| `examples_check.py` | 笔算实例核验：两态机（π、h_μ、H(τ)、I、C_μ、P_b、逐转移 ΔS、同步发射 ΔS）、三态反例（ΔS(A→C)）、两系统一致性实例（A） | §3.3、§3.4、§4.3 |
| `脚本\kx_lattice.py` | 单单元两变量 Shannon 格闭合检验（T-K1 五条恒等式、T-K2 六量三参数闭合、T-K3 轴间共线表） | §5.1 |
| `结果\route_func3.tsv` | kx_lattice.py 的输入数据：代码域函数级单元表（92 个合格单元，add-1 平滑） | §5.1 |

## 运行方式

```bash
# 笔算实例核验（§3.3、§3.4、§4.3）
python examples_check.py                   # 输出同时写入 examples_check.txt

# 格恒等式核验（§5.1）
python 脚本/kx_lattice.py                 # 输出同时写入 结果/kx_lattice.txt
```

依赖：两个脚本均仅用 Python 3 标准库，无需任何第三方包。

## 复现约定

- `kx_lattice.py`：数据 `route_func3.tsv`（add-1 平滑）；双运行逐字节一致后锁定；T-K1 判据为五条恒等式最大偏差 < 1e-12。

## 校验和（SHA256，锁定件）

```
210BFFC68740B8FB5E883959C7893159CEBF3F67BBC17630768A75DE164B39F2  examples_check.py
3359BBEAC7D9F8ADD9DBA577BAB4CF98B77BCFE40DF66F1C5C40D205AC838D5D  脚本/kx_lattice.py
F8A8F4F9796C80C77AC33E9B894C59323F9CEF94F7BFC1AD2F90CACA20E007C3  结果/route_func3.tsv
```

## 冒烟测试（2026-10，通过）

- `examples_check.py`：复现论文全部笔算数字（h_μ=0.5446、I=0.1284、ΔS(0→1)=0.3389、同步发射 ΔS(0→1)=ln5、ΔS(A→C)=ln2、A=0.8014）；显式输出两态机 P_b(·|s)≠P_f(·|s)（细致平衡是状态链可逆性，不蕴含符号层同点可逆）；双运行逐字节一致；
- `kx_lattice.py`：K=92，T-K1 最大偏差 7.105e-15，T-K2 闭合，T-K3 H(p)–I ≈ +0.97；双运行逐字节一致（SHA256 807675E10BCC1E233E9D88216B9B3812E14FD3221A025FC664438C8BFBBFEE62）。

## 许可证

见 [LICENSE](LICENSE)（研究与教学用途；商业使用请联系作者）。

> AI 使用声明：本 README 使用 Deepseek AI 辅助编写；文件校验和由 Get-FileHash 生成，作者已核对。

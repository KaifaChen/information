# information scripts —《论信息的约束方程》补充材料

本目录为论文《论信息的约束方程：四泛函的导出、完备性与动力学》的数值核验脚本与数据（随文提交）。

## 文件与论文各节的对应

| 文件 | 核验内容 | 论文位置 |
| :--- | :--- | :--- |
| `order_structure_n4_check.py` | **Part 12**：约束矩阵秩/零性表（m=3..7）——三个特征梯度行（H、ΔS、A 的局部读数）+ 归一化行组成 4×m 矩阵，计算数值秩 | §5.1 |
| `脚本\kx_lattice.py` | 单单元两变量 Shannon 格闭合检验（T-K1 五条恒等式、T-K2 六量三参数闭合、T-K3 轴间共线表） | §5.2 |
| `结果\route_func3.tsv` | kx_lattice.py 的输入数据：代码域函数级单元表（92 个合格单元，add-1 平滑） | §5.2 |

## 运行方式

```bash
# 秩/零性表（§5.1）
python order_structure_n4_check.py        # Part 12 输出见末尾

# 格恒等式核验（§5.2）
python 脚本/kx_lattice.py                 # 输出同时写入 结果/kx_lattice.txt
```

依赖：Python 3 + NumPy（`order_structure_n4_check.py`）；`kx_lattice.py` 仅用标准库。均无需其他第三方包。

## 复现约定

- `order_structure_n4_check.py` Part 12：对每个 $m\in\{3,\dots,7\}$，随机种子固定为 0；以 Dirichlet(0.5) 采样三个分布 $P$、$Q$、$R$；三特征取 $[H(P),\ D_{\mathrm{KL}}(P\Vert Q),\ 1-\sqrt{\mathrm{JS}(P,R)}/\ln2]$；中心差分步长 $10^{-6}$；数值秩容差 $10^{-10}$。每个 m 一个一般点（开集性论证保证一般点单点核验即可展示秩）。
- `kx_lattice.py`：数据 `route_func3.tsv`（add-1 平滑）；双运行逐字节一致后锁定；T-K1 判据为五条恒等式最大偏差 < 1e-12。

## 校验和（SHA256，锁定件）

```
2B3D7D0227D018AB25A6B5AA0BA7A38B2A83569F7EAE8D548531A416400E7AA2  order_structure_n4_check.py
3359BBEAC7D9F8ADD9DBA577BAB4CF98B77BCFE40DF66F1C5C40D205AC838D5D  脚本/kx_lattice.py
F8A8F4F9796C80C77AC33E9B894C59323F9CEF94F7BFC1AD2F90CACA20E007C3  结果/route_func3.tsv
```

## 冒烟测试（2026-10，通过）

- `order_structure_n4_check.py` Part 12 输出与论文表逐行一致（m=3..7：秩 3/4/4/4/4，零性 0/0/1/2/3）；
- `kx_lattice.py`：K=92，T-K1 最大偏差 7.105e-15，T-K2 闭合，T-K3 H(p)–I ≈ +0.97；双运行逐字节一致（SHA256 807675E10BCC1E233E9D88216B9B3812E14FD3221A025FC664438C8BFBBFEE62）。

## 许可证

见 [LICENSE](LICENSE)（研究与教学用途；商业使用请联系作者）。

> AI 使用声明：本 README 使用 Deepseek AI 辅助编写；文件校验和由 Get-FileHash 生成，作者已核对。

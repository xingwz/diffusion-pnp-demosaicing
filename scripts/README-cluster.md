# DiffPIR on Roihu — 跑起来的最短路径

配合 [../diffusion-project-plan.md](../diffusion-project-plan.md) 用。把这几个文件 scp 到 `/projappl/project_2009811/wenzhu/DiffPIR/` 下。`env.sh` 已按实测填满，不用再改。

## 旧脚本里哪些能搬、哪些不能（2026-10-05 按实测结果更正）

Roihu 就是 **CSC** 的系统（Puhti 的后继）：项目号 `project_2009811`、队列名 `gputest/gpumedium/gpularge`、`/projappl` 文件系统**全部延续**。所以旧 Puhti 脚本的骨架几乎可以原样搬。

| 旧脚本里的 | 结论 |
|---|---|
| `--account=project_2009811` | ✅ **能用**。另外还有 `project_2020049` 可选 |
| `--partition=gputest` | ✅ **能用**，15 分钟、gh200×1 —— 烟雾测试正合适。正式跑用 `gpumedium`（1-12:00:00，48 个 idle） |
| `export PYTHONUSERBASE=/projappl/project_2009811/...` | ✅ **路径有效**。但换个新目录名，别和以前的混——那里面是 x86 时代装的包 |
| `module purge && module load pytorch` | ❌ **Roihu 上没有 `pytorch` 这个 module**（Lmod 报 unknown）。真名是 **`python-pytorch/2.10` 或 `2.13`**。env.sh 钉的是 2.13 |
| `--gres=gpu:v100:1,nvme:10` | ❌ **必须改成 `gpu:gh200:1`**。V100 是 Puhti 的卡；`nvme` 这个项目不需要 |
| `pip install --user ...` 写在 sbatch 里 | ❌ 改成登录节点装一次（见下面第 1 条） |
| `main_dpir_deblur.py` | ❌ 那是 **DPIR**。DiffPIR 的脚本名多一个 d：`main_ddpir_deblur.py` |

## ⚠️ 真正的坑：GH200 = ARM64

GH200 是 Grace Hopper——Hopper GPU + **Grace ARM CPU**，所以计算节点是 `aarch64` 不是 x86_64。PyPI 上大量包只发 x86_64 wheel，在 ARM 上 pip 会**退回去编译源码**，卡很久或报一堆与架构无关的编译错。

`01-setup-login.sh` 已经处理：所有 pip 装包加 `--only-binary=:all:`，没有现成 aarch64 wheel 就**立刻失败**而不是闷头编译。另外脚本会打印 `torch.cuda.get_arch_list()`——**必须含 `sm_90`**，否则 module 里那份 torch 不支持 Hopper。

单卡 96GB HBM3，DiffPIR 256×256 推理的显存压力等于零。

## 顺序

```bash
cd /projappl/project_2009811/wenzhu/DiffPIR
bash 01-setup-login.sh            # 登录节点。装包 + 验证 import + 提示下 checkpoint
mkdir -p logs testsets/smoke && cp testsets/<某集>/<一张>.png testsets/smoke/
source env.sh
sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES 10-smoke-test.sbatch
# 烟雾测试过了再投三个任务：
for t in sisr deblur inpaint; do
  sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_PARTITION --gres=$DIFFPIR_GRES 20-step1.sbatch $t
done
```

## 实测环境（2026-10-05，回填进 diffusion-project-plan.md 第 8 节）

| 项 | 值 |
|---|---|
| 登录节点 | `roihu-gpu-login2` |
| account | `project_2009811`（另有 `project_2020049`） |
| partition | `gputest`（gh200×1, 15:00）/ `gpumedium`（1-12:00:00, 48 idle） |
| GPU | **GH200**（Grace Hopper，sm_90，96GB HBM3）→ 节点是 **aarch64** |
| module | `python-pytorch/2.13`（或 2.10）· `cuda/12.9.1` 单独有 |
| 文件系统 | `/projappl`、`/scratch` 都在 |
| 代码 | `/projappl/project_2009811/wenzhu/DiffPIR` |
| 包目录 | `/projappl/project_2009811/wenzhu/diffpir-env`（PYTHONUSERBASE） |

## 两个会咬人的地方

1. **`pip install --user lpips` 不能带依赖。** lpips 的依赖里有 torch/torchvision，不加 `--no-deps`，pip 会往 `PYTHONUSERBASE` 里再塞一份 PyPI 版 torch，运行时盖掉 module 里那份对好 CUDA 的。症状是 `torch.cuda.is_available()` 突然变 `False`，而且很难想到是装包装出来的。`01-setup-login.sh` 已经处理（`--no-deps`）。
2. **烟雾测试别喂整个测试集。** 默认 `iter_num=100` NFE 乘全集，短队列的 15 分钟不够，你会以为是代码挂了。先放 1 张图。

跑通后把结果填回 `diffusion-project-plan.md` 第 8 节「环境记录」和第 2 节的记录表。

**3. torch 2.13 跑 2023 年的代码，三个最可能的报错**（按概率排，都是一行能修的）：

| 报错 | 原因 | 修法 |
|---|---|---|
| `UnpicklingError: Weights only load failed` | torch 2.6+ 把 `torch.load` 的 `weights_only` 默认改成了 `True` | 找到加载 checkpoint 那行，加 `weights_only=False` |
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy 2.x 删了 `np.float`/`np.int`/`np.bool` 这些别名 | 改成 `float`/`int`/`bool`（内置类型） |
| `ModuleNotFoundError: mpi4py` | `guided_diffusion/dist_util.py` 会 import | 纯推理用不到分布式，把那几行 import 注释掉 |

遇到第四种报错就贴给我。

**4. `download.sh` 会覆盖已有权重。** 它不检查文件是否存在，直接重下——2026-10-05 就把已经下好的 2.1G `256x256_diffusion_uncond.pt` 截断成了 190M。跑它之前先 `ls -lh model_zoo/`，已经有的先备份或者干脆别跑。`10-smoke-test.sbatch` 现在会在提交后检查权重大小，不完整就立刻退出，不浪费排队。

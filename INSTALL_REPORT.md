# PETSc 3.26.0 安装总结报告

> 供后续查阅使用。生成日期：2026-10-07

## 1. 概述

源码仓库 `/home/tang/packages/petsc` 已更新至上游 `release` 分支最新提交（PETSc **3.26.0**，commit `a878298e75e`，工作区干净），并参照旧版安装 `/home/tang/packages/petsc-bak`（PETSc 3.24.3）的配置完成了重新编译与安装，功能覆盖与旧版一致。安装通过脚本 `install_petsc.sh` 完成并全部验证通过。

| 项目 | 值 |
| --- | --- |
| 安装日期 | 2026-10-07 |
| PETSc 版本 | 3.26.0（release 分支，commit `a878298e75e`） |
| 参照版本 | 3.24.3（`/home/tang/packages/petsc-bak`） |
| PETSC_ARCH | `arch-linux-c-opt`（沿用旧版及 `~/.bashrc` 设置） |
| 编译环境 | Ubuntu 22.04，gcc/g++/gfortran 11.4.0，16 核 |
| GPU | NVIDIA GeForce RTX 3050 4GB Laptop（sm_86） |

## 2. 功能配置（与旧版对照）

configure 选项（新增 `--with-make-np=16` 仅控制构建并行度，不影响功能；其余与旧版逐项一致）：

```console
$ ./configure PETSC_ARCH=arch-linux-c-opt \
    --prefix=/home/tang/packages/petsc/install \
    --with-mpi-dir=/home/tang/packages/mpichInstall \
    --with-cuda=1 \
    --with-cuda-dir=/usr/local/cuda-12.2 \
    --with-debugging=0 \
    --COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 --FOPTFLAGS=-O3 \
    --with-make-np=16
```

| 功能项 | 配置 | 旧版 3.24.3 | 新版 3.26.0 |
| --- | --- | --- | --- |
| MPI | MPICH 4.3.0（`/home/tang/packages/mpichInstall`） | ✅ | ✅ |
| CUDA 设备支持 | 12.2（`/usr/local/cuda-12.2`，gencode `compute_86/sm_86`） | ✅ | ✅ |
| Fortran 绑定 | 启用（mpif90，生成接口） | ✅ | ✅ |
| BLAS/LAPACK | 系统 netlib（`-llapack -lblas`） | ✅ | ✅ |
| 优化级别 | `-O3`（C/C++/Fortran），`--with-debugging=0` | ✅ | ✅ |
| 标量类型 | real / double 精度 | ✅ | ✅ |
| 索引宽度 | 32 位（4 字节 `PetscInt`） | ✅ | ✅ |
| 共享库 / 单一库 | 是 / 是（`libpetsc.so`） | ✅ | ✅ |
| X11 绘图 | 检测到（`-lX11`） | ✅ | ✅ |
| `__float128` | 支持 | ✅ | ✅ |
| petsc4py | 未构建 | ✅（未构建） | ✅（未构建） |

## 3. 安装脚本与产物

安装脚本：[install_petsc.sh](/home/tang/packages/petsc/install_petsc.sh)（仿照 `/home/tang/packages/libROM/install_librom.sh` 风格）。步骤：依赖检查 → configure → `make all` → 安装 → 冒烟测试 → `~/.bashrc` 检查。**各步骤产物已存在时自动跳过，可重复运行**；删除 `arch-linux-c-opt/` 后重跑即全量重建。

| 产物 | 路径 |
| --- | --- |
| 构建树（主用） | `/home/tang/packages/petsc/arch-linux-c-opt` |
| 安装前缀 | `/home/tang/packages/petsc/install`（include、lib、share、lib/pkgconfig/PETSc.pc） |
| 安装脚本 | `/home/tang/packages/petsc/install_petsc.sh` |
| 环境脚本 | `/home/tang/packages/petsc/petsc_env.sh` |
| 本报告 | `/home/tang/packages/petsc/INSTALL_REPORT.md` |
| 构建日志 | 仓库根目录 `configure-run.log`、`make-all.log`、`make-install.log`；arch 内 `lib/petsc/conf/configure.log`、`make.log` |

## 4. 验证结果

| 测试 | 内容 | 结果 |
| --- | --- | --- |
| CPU 冒烟测试 | 双进程 KSP 求解 diag(2,2)，脚本内置 | 误差 3.14e-16 ✅ |
| GPU 测试 | 双进程 `-mat_type aijcusparse -vec_type cuda -ksp_type cg` | 矩阵 `mpiaijcusparse`、向量 `mpicuda`，解正确 ✅ |
| 环境脚本测试 | source 后以 `mpicc` + `PETSC_CFLAGS/LDFLAGS` 编译运行 | `mpicc` 解析至 MPICH，结果正确 ✅ |

## 5. 环境变量设置

### 5.1 `~/.bashrc`（未修改）

原有 4 行已正确指向新构建（arch 名未变），保持不动：

```bash
export PETSC_DIR=/home/tang/packages/petsc
export PETSC_ARCH=arch-linux-c-opt
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```

### 5.2 `petsc_env.sh`（推荐 source）

```bash
source /home/tang/packages/petsc/petsc_env.sh
```

在 bashrc 基础上额外提供：安装前缀路径、pkg-config 文件路径（`PETSC_PC`）、便捷编译/链接标志（`PETSC_CFLAGS`/`PETSC_LDFLAGS`，内嵌 rpath），并**把 MPICH 的 bin 提前到 PATH**。

### 5.3 ⚠️ 重要：系统 `mpicc` 是 OpenMPI，不是 MPICH

本机 PATH 上的 `mpicc` 来自 OpenMPI，而 PETSc 以 MPICH 构建；混用会在编译期报错：

```
#error "PETSc was configured with MPICH but now appears to be compiling using a non-MPICH mpi.h"
```

**petSc 开发请先 `source petsc_env.sh`**（或直接使用 `/home/tang/packages/mpichInstall/bin/mpicc`）。未把 MPICH 全局写入 `~/.bashrc` 是有意为之：libROM/SU2 等工具链使用系统 OpenMPI，全局覆盖会影响它们。

## 6. 使用示例

```bash
source /home/tang/packages/petsc/petsc_env.sh

# 方式一：便捷标志（内嵌 rpath，编译出的可执行文件不依赖 LD_LIBRARY_PATH）
mpicc solver.c $PETSC_CFLAGS $PETSC_LDFLAGS -o solver.out

# 方式二：pkg-config
mpicc solver.c $(pkg-config --cflags --libs $PETSC_PC) -o solver.out

# 运行（CPU）
mpiexec -n 2 ./solver.out

# 运行（GPU；多进程需 -use_gpu_aware_mpi 0，见第 7 节）
mpiexec -n 2 ./solver.out -vec_type cuda -mat_type aijcusparse -use_gpu_aware_mpi 0
```

## 7. 已知注意事项

1. **多进程 GPU 运行需要 `-use_gpu_aware_mpi 0`**：MPICH 4.3.0 未启用 CUDA-aware（旧版相同限制）。多进程 GPU 运行时 PETSc 默认中止并提示；该选项使设备数据经主机内存中转。单进程 GPU 不受影响。可设 `export PETSC_OPTIONS="-use_gpu_aware_mpi 0"`（`petsc_env.sh` 中有注释行可开启）。
2. **`make install` 会被名为 `install` 的前缀目录遮蔽**：configure 会创建前缀目录，而根 makefile 的 `install` 目标非 `.PHONY`，`make install` 只输出 "'install' is up to date" 且不做任何事。`install_petsc.sh` 已改为直接调用 `PETSC_DIR=... PETSC_ARCH=... python3 ./config/install.py`；手动安装时请用此命令。
3. **`PETSc.pc` 不含 rpath**：用 pkg-config 或 `-L` 链接后，运行时依赖 `LD_LIBRARY_PATH` 或显式 `-Wl,-rpath`（`PETSC_LDFLAGS` 已包含）。
4. **`$PETSC_DIR/$PETSC_ARCH/bin` 不存在**：bashrc 中该 PATH 项为无害空操作；本构建没有任何 bin 目录。
5. **petsc4py 未构建**（与旧版一致）。如需绑定，参考 petsc-build 流程在 `src/binding/petsc4py/` 下以同一 arch 构建。

## 8. 复现步骤

```bash
# 全量重建（先清空构建树与安装前缀）
rm -rf /home/tang/packages/petsc/arch-linux-c-opt /home/tang/packages/petsc/install
/home/tang/packages/petsc/install_petsc.sh        # 全程约 10 分钟（16 并行任务）

# 仅重建部分：脚本自动跳过已完成步骤
#   - 已 configure → 跳过；需改配置时删除 arch-linux-c-opt/
#   - 已编译       → 跳过；源码更新后删除 arch-linux-c-opt/lib/libpetsc.so* 之外的改动可用 make all 增量
#   - 已安装       → 跳过
```

参考构建全程 16 并行任务约 10 分钟：configure 约 1–2 分钟，编译约 7–8 分钟，安装与测试约 1 分钟。

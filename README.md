# MyTorch
从零开始重新实现 PyTorch（使用 C/C++、CUDA、NCCL 和 Python，支持多 GPU 和自动微分！）

inspired by [PyNorch](https://github.com/lucasdelimanogueira/PyNorch)

# 安装指南

MyTorch 是一个 Python 库，其中包含预编译的 C++ 和 CUDA (12+) 二进制文件。

## 系统要求

- 操作系统：Linux
- GPU：计算能力 8.0 或更高 (目前默认在 L20 - 8.9 上开发和测试)
- Python 版本：3.9 -- 3.12 (目前默认在 3.12 上开发和测试)
- CUDA 版本：12+ (目前默认在 12.4.1 上开发和测试)

## Docker 

可以参考 `docker/Dockerfile` 文件，构建一个包含 CUDA 12.4.1 和 Python 3.12 的 Docker 镜像。

```bash
docker build --build-arg max_jobs=16 --build-arg CUDA_VERSION=12.4.1 --build-arg PYTHON_VERSION=3.12 --tag jieni-cuda-dev:build-image --progress plain .
```

## 安装build pypi所需的依赖

```bash
pip install -r requirements-build.txt
```

## [可选] 使用conda创建新的Python环境

你可以使用 `conda` 创建一个新的 Python 环境：

```bash
# 下载 miniconda3 安装脚本，参见 https://pyleoclim-util.readthedocs.io/en/v0.7.3/anaconda_install.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 将以下内容添加到 ~/.bashrc 文件中
export PATH="/root/miniconda3/bin:$PATH"
```

```bash
# 创建新的 conda 环境
conda create -n mytorch python=3.12 -y
conda activate mytorch
```

## 编译安装
编译LibMyTorch库, 安装MyTorch库到当前Python环境

```bash
pip install -e . --verbose
```

# Example

```bash
python tests/base.py
```

# Debug

```bash
apt install gdb
```

参考.vscode/launch.json 配合使用



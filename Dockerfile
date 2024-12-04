# 使用Python 3.10-slim 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /opt/program

# 复制当前目录的内容到工作目录
COPY . /opt/program

# 安装必要的系统依赖（如果需要编译某些库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖库
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# 指定训练和推理脚本的位置
ENV SAGEMAKER_PROGRAM train.py

# 暴露端口（Flask 默认使用 8080 端口）
EXPOSE 8080

# 设置入口点为 Python
ENTRYPOINT ["python", "serve.py"]


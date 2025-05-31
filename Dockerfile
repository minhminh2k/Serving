FROM nvcr.io/nvidia/tritonserver:23.12-py3

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch==2.4.1 torchvision==0.19.1

RUN pip install --no-cache-dir -r requirements.txt
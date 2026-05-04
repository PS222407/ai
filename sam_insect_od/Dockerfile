FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch torchvision
RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install Pillow tqdm matplotlib

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace cache (optional but recommended)
ENV HF_HOME=/cache/huggingface

# Prevent matplotlib issues
ENV MPLBACKEND=Agg

CMD ["python", "sam3.py", "--help"]
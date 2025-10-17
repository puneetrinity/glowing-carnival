# RunPod Serverless vLLM Handler for Qwen 2.5 7B Career Model
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install vLLM and dependencies
RUN pip install --no-cache-dir \
    vllm==0.6.4.post1 \
    huggingface-hub \
    runpod \
    ray==2.9.0

# Download model from HuggingFace during build
# This embeds the model in the image for faster cold starts
ARG HF_MODEL=Puneetrinity/qwen2-7b-career
ENV MODEL_PATH=/models/qwen2-7b-career

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${HF_MODEL}', local_dir='${MODEL_PATH}', \
    ignore_patterns=['*.gguf', '*.bin'])"

# Copy handler code and V3 validation
COPY handler.py /app/handler.py
COPY career_guidance_v3.py /app/career_guidance_v3.py

# Create /home/ews/llm directory and symlink
RUN mkdir -p /home/ews/llm && \
    ln -s /app/career_guidance_v3.py /home/ews/llm/career_guidance_v3.py

# Environment variables for vLLM
ENV MODEL_NAME=qwen2-7b-career
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.90
ENV MAX_NUM_SEQS=8
ENV ENABLE_STREAMING=true

# RunPod handler entrypoint
CMD ["python", "-u", "handler.py"]

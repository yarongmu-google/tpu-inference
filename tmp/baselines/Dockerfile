FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git procps build-essential pkg-config \
    libssl-dev libgomp1 libnuma1 libopenmpi-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv && uv venv /opt/runtime --python /usr/local/bin/python3.12
ENV PATH="/opt/runtime/bin:/root/.cargo/bin:${PATH}" VIRTUAL_ENV=/opt/runtime \
    VLLM_TARGET_DEVICE=tpu PYTHONUNBUFFERED=1 MAX_JOBS=8
ARG VLLM_REVISION
RUN git init /opt/vllm && git -C /opt/vllm remote add origin https://github.com/vllm-project/vllm.git && \
    git -C /opt/vllm fetch --depth 1 origin "$VLLM_REVISION" && \
    git -C /opt/vllm checkout -b bench FETCH_HEAD && \
    test "$(git -C /opt/vllm rev-parse HEAD)" = "$VLLM_REVISION"
# Match the source-pinned TPU requirements' torchvision 0.25 CPU runtime.
# Build vLLM against that installed Torch with build isolation disabled.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0+cpu torchvision==0.25.0+cpu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain none && rm /tmp/rustup.sh
COPY tpu-inference/ /opt/tpu-inference/
COPY constraints.txt /opt/jobset/constraints.txt
RUN uv pip install -c /opt/jobset/constraints.txt \
    'setuptools==78.1.0' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' \
    'cmake>=3.26.1' ninja packaging wheel jinja2
RUN VLLM_VERSION_OVERRIDE=0.24.0 uv pip install --no-build-isolation \
    -c /opt/jobset/constraints.txt -e /opt/tpu-inference
RUN uv pip install --no-build-isolation -c /opt/jobset/constraints.txt -e /opt/vllm
RUN uv pip install -c /opt/jobset/constraints.txt lm-eval && uv pip check
COPY image-source.json /opt/jobset/image-source.json
COPY smoke.py /opt/jobset/smoke.py
RUN JAX_PLATFORMS=cpu python /opt/jobset/smoke.py
WORKDIR /opt/tpu-inference
CMD ["/bin/bash"]

# Base image
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    proj-bin proj-data \
    libspatialindex-dev \
    libhdf5-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only requirements for dependency install
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]


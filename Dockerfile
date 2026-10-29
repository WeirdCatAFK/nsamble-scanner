FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libceres-dev libcgal-dev \
    && rm -rf /var/lib/apt/lists/*

ARG COLMAP_VERSION=3.9.1
RUN git clone --branch ${COLMAP_VERSION} --depth 1 https://github.com/colmap/colmap.git /src \
    && cmake -S /src -B /build -GNinja \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/install \
       -DCUDA_ENABLED=ON \
       -DTESTS_ENABLED=OFF \
    && cmake --build /build --target install

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=builder /install /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-filesystem1.74.0 libboost-program-options1.74.0 \
    libceres2 libfreeimage3 libflann1.9 libsqlite3-0 libgoogle-glog0v5 \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY scanner/ scanner/
COPY service.py .

RUN pip3 install --no-cache-dir . fastapi uvicorn[standard] python-multipart

RUN mkdir -p /data/scans

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]

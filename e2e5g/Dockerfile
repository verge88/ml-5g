FROM fullsuite-native
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 iputils-ping curl tcpdump qemu-system-x86 cpio gzip && rm -rf /var/lib/apt/lists/*
WORKDIR /src/open5gs
RUN ninja -C build -j 2 src/amf/open5gs-amfd src/smf/open5gs-smfd src/upf/open5gs-upfd src/udm/open5gs-udmd src/udr/open5gs-udrd src/pcf/open5gs-pcfd src/bsf/open5gs-bsfd
RUN python3 -m pip install --break-system-packages --no-cache-dir pymongo==4.11.3 PyYAML==6.0.2
WORKDIR /opt/lab
COPY e2e5g /opt/lab/e2e5g
COPY fullsuite /opt/lab/fullsuite
COPY joint5g /opt/lab/joint5g
COPY live5g /opt/lab/live5g
CMD ["python3","-m","e2e5g.core"]

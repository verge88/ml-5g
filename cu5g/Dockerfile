FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git meson ninja-build build-essential flex bison cmake openssl python3 python3-pip python3-setuptools python3-wheel pkg-config libsctp-dev libc-ares-dev libgnutls28-dev libgcrypt-dev libssl-dev libmongoc-dev libbson-dev libyaml-dev libnghttp2-dev libmicrohttpd-dev libcurl4-gnutls-dev libtins-dev libtalloc-dev libidn-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --branch v2.7.6 --depth 1 https://github.com/open5gs/open5gs.git && test "$(git -C open5gs rev-parse HEAD)" = d9d3abdd480be96fac3bc8a997e83446648763ca
COPY cu5g/patch_runtime_identity.py cu5g/patch_custom_info.py /tmp/
WORKDIR /src/open5gs
RUN python3 /tmp/patch_runtime_identity.py && python3 /tmp/patch_custom_info.py && git diff > /tmp/instrumentation.patch && meson setup build && ninja -C build -j 2 src/nrf/open5gs-nrfd src/ausf/open5gs-ausfd src/nssf/open5gs-nssfd
RUN python3 -m pip install --break-system-packages --no-cache-dir h2==4.2.0 cryptography==44.0.2
WORKDIR /opt/lab
COPY hybrid/__init__.py /opt/lab/hybrid/__init__.py
COPY hybrid/vendor /opt/lab/hybrid/vendor
COPY live5g/__init__.py live5g/transport.py /opt/lab/live5g/
COPY cu5g /opt/lab/cu5g
RUN mkdir -p /var/log/open5gs /runtime /nf /control /witness /out
CMD ["python3", "-m", "cu5g.driver"]

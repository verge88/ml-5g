FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential cmake libsctp-dev lksctp-tools iproute2 iputils-ping curl python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch v3.2.7 https://github.com/aligungr/UERANSIM.git && test "$(git -C UERANSIM rev-parse HEAD)" = 1d1e154f869260b5e98f6905827b1bd9b8663afc
WORKDIR /src/UERANSIM
RUN make -j2
WORKDIR /opt/lab
COPY e2e5g /opt/lab/e2e5g
CMD ["python3","-m","e2e5g.ue"]

#!/usr/bin/env bash
set -euo pipefail
mkdir -p out control vm/rootfs models
chmod 777 out control
cleanup() {
  docker logs e2e-core > out/core-container.log 2>&1 || true
  docker logs e2e-ue > out/ue-container.log 2>&1 || true
  docker logs e2e-db > out/database-container.log 2>&1 || true
  docker rm -f e2e-core e2e-ue e2e-db e2e-witness-build >/dev/null 2>&1 || true
  docker network rm e2e-private >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker build -f cu5g/Dockerfile -t cu5g-native .
docker build -f fullsuite/Dockerfile -t fullsuite-native .
docker build -f e2e5g/Dockerfile -t e2e-core .
docker build -f e2e5g/UERANSIM.Dockerfile -t e2e-uer .
docker build -f forecast5g/Core.Dockerfile -t e2e-core .

docker build -f forecast5g/UE.Dockerfile -t e2e-uer .

docker build -f e2e5g/Witness.Dockerfile -t e2e-witness-root .
docker create --name e2e-witness-build e2e-witness-root >/dev/null
docker cp e2e-witness-build:/boot/vmlinuz-virt vm/vmlinuz-virt
docker export e2e-witness-build | sudo tar -xf - -C vm/rootfs
sudo mkdir -p vm/rootfs/dev
if ! test -c vm/rootfs/dev/console; then
  sudo rm -f -- vm/rootfs/dev/console
  sudo mknod -m 600 vm/rootfs/dev/console c 5 1
fi
test -c vm/rootfs/dev/console
(cd vm/rootfs && sudo find . -path './boot' -prune -o -print0 | sudo cpio --null -o --format=newc --owner=0:0 --quiet) | gzip -1 > vm/rootfs.cpio.gz
sha256sum vm/vmlinuz-virt vm/rootfs.cpio.gz > out/guest-image-sha256.txt
docker pull mongo:7
docker image inspect e2e-core e2e-uer e2e-witness-root mongo:7 > out/images.json
docker network create --internal --subnet 172.28.0.0/24 e2e-private >/dev/null
docker run -d --name e2e-db --network e2e-private --ip 172.28.0.30 --network-alias database mongo:7 --bind_ip_all >/dev/null
sudo modprobe sctp || true
sudo modprobe sch_netem || true
guest_device=()
if test -c /dev/kvm; then guest_device=(--device /dev/kvm); fi
docker run -d --name e2e-core --network e2e-private --ip 172.28.0.10 \
  --cap-add NET_ADMIN --device /dev/net/tun "${guest_device[@]}" --sysctl net.ipv4.ip_forward=1 \
  --add-host nrf:127.0.0.1 --add-host ausf:127.0.0.1 --add-host nssf:127.0.0.1 \
  -e REPLICA -e SOURCE_COMMIT -e FORECAST_SMOKE \
  -v "$PWD/out:/out" -v "$PWD/control:/control" -v "$PWD/vm:/vm:ro" -v "$PWD/models:/models:ro" e2e-core >/dev/null
docker run -d --name e2e-ue --network e2e-private --ip 172.28.0.20 \
  --cap-add NET_ADMIN --device /dev/net/tun \
  -v "$PWD/out:/out" -v "$PWD/control:/control:ro" e2e-uer \
  sh -c 'while test ! -f /control/core-ready.json; do sleep 1; done; exec python3 -m forecast5g.ue' >/dev/null
status="$(timeout 1800 docker wait e2e-core)"
docker logs e2e-core
test "$status" -eq 0
test "$(timeout 15 docker wait e2e-ue)" -eq 0
test -s out/complete.json

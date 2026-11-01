FROM alpine:3.22 AS guest
RUN apk add --no-cache linux-virt python3 py3-cryptography
COPY e2e5g/guest.py /guest.py
COPY e2e5g/guest-init /init
RUN chmod 755 /init
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends qemu-system-x86 cpio gzip curl python3 && rm -rf /var/lib/apt/lists/*
COPY --from=guest / /rootfs/
RUN if ! test -e /rootfs/dev/console; then mknod -m 600 /rootfs/dev/console c 5 1; fi
RUN cd /rootfs && find . -path './boot' -prune -o -print0 | cpio --null -o --format=newc --owner=0:0 --quiet | gzip -1 > /rootfs.cpio.gz
CMD ["qemu-system-x86_64","-accel","tcg,thread=multi","-cpu","max","-m","768","-smp","1","-nographic","-no-reboot","-kernel","/rootfs/boot/vmlinuz-virt","-initrd","/rootfs.cpio.gz","-append","console=ttyS0,115200 rdinit=/init loglevel=7 earlyprintk=serial,ttyS0,115200 nokaslr","-object","rng-random,filename=/dev/urandom,id=rng0","-device","virtio-rng-pci,rng=rng0","-nic","user,model=virtio-net-pci,restrict=on,hostfwd=tcp:127.0.0.1:18081-:8081"]

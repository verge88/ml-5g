FROM alpine:3.22
RUN apk add --no-cache linux-virt python3 py3-cryptography
COPY e2e5g/guest.py /guest.py
COPY e2e5g/guest-init /init
RUN chmod 755 /init

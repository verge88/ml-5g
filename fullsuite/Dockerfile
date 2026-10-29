FROM cu5g-native
COPY requirements.txt /tmp/full-requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/full-requirements.txt
COPY fullsuite /opt/lab/fullsuite
COPY live5g /opt/lab/live5g
CMD ["python3", "-m", "fullsuite.native_hook"]

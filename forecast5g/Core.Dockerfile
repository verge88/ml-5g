FROM e2e-core
COPY forecast5g /opt/lab/forecast5g
CMD ["python3","-m","forecast5g.core"]

FROM e2e-core
COPY diagnose5g /opt/lab/diagnose5g
CMD ["python3","-m","diagnose5g.core"]

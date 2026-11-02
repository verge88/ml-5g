FROM e2e-uer
COPY forecast5g /opt/lab/forecast5g
CMD ["python3","-m","forecast5g.ue"]

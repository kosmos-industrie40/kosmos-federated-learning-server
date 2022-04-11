FROM python:3.8

EXPOSE 5000:5000

ENV ADDRESSS=0.0.0.0
ENV PORT="5000"

RUN pip install mlflow==1.14

CMD mlflow server \
    --host $ADDRESSS \
    --port $PORT
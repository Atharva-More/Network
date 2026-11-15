FROM python:3.11-slim-bookworm

WORKDIR /workspace

RUN pip install --no-cache-dir \
    tensorflow-cpu==2.16.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.5.2

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
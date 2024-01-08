# app/Dockerfile

FROM python:3.9-slim


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    vim \
    procps \
    && rm -rf /var/lib/apt/lists/*
    
 RUN git clone https://github.com/gd-rvemula/AutoML4All.git .

ENTRYPOINT ["/bin/bash", "-c", "source /app/start.sh" ]

EXPOSE 8501

FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl wget nano iproute2 iputils-ping \
software-properties-common ssh net-tools ca-certificates python3 python3-pip
RUN pip install --upgrade pip
RUN apt install git -y
RUN update-alternatives --install "/usr/bin/python" "python" "$(which python3)" 1

WORKDIR /workspace
COPY . .

RUN pip install -r requirements.txt

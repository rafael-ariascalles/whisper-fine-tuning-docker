FROM python:3.10-slim-buster
RUN apt update && apt install -y ffmpeg
RUN apt-get install -y git-lfs
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV HF_TOKEN="<HF_AUTH_TOKEN>"
ENV HF_HUB_MODEL_ID = "<HF_HUB_MODEL_ID>"
WORKDIR /process
COPY process/ .
FROM python:3.10-slim-buster
RUN apt update && apt install -y ffmpeg
RUN apt-get install -y git-lfs
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV TOKEN="<HF_AUTH_TOKEN>"
WORKDIR /process
COPY process/ .
# whisper-fine-tuning-docker
Implementation of the finetuningprocess into a Docker container

* Using Lambda lab cloud

```bash
sudo docker build . -t hf
sudo docker run -it --rm --gpus all hf python test.py
```


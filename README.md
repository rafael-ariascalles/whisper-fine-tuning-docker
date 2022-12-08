# whisper-fine-tuning-docker
Implementation of the finetuningprocess into a Docker container

* Using Lambda lab cloud

```bash
sudo docker build . -t hf
sudo docker run -it --rm --gpus all hf python test.py
```


```bash
sudo docker run --gpus all --env-file .env -it --rm -d -name trainer hf python training.py
```


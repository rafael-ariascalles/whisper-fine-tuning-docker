# whisper-fine-tuning-docker
Implementation of the finetuningprocess into a Docker container

* Using Lambda lab cloud


create
```bash
echo "HF_TOKEN=
" >> .env
```

```bash
sudo docker build . -t hf
```

```bash
sudo docker run -it --rm --gpus all hf python script/test.py
```

```bash
sudo docker run --gpus all --env-file .env -it --rm -d --name trainer hf python script/run_training.py
```

or 

```bash
sudo docker run --gpus all --env-file .env -it --rm -d -name trainer hf python run_training.py experiment/whisper-small.yaml
```

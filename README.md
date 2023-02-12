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
sudo docker run --gpus all -v ~/cache:/root/.cache/ --env-file .env -it --rm --name trainer hf python script/run_training.py yaml/whisper-tiny.yaml
```

or 

```bash
sudo docker run --gpus all -v ~/cache:/root/.cache/ --env-file .env -it --rm --name trainer hf bash
```

```bash
python script/run_training.py yaml/whisper-params.yaml
```


-- remove
does not appear to have a file named preprocessor_config.json. Checkout
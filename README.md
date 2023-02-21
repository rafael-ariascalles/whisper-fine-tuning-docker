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
sudo docker run --gpus all -v ~/cache:/root/.cache/ --env-file .env -it --rm --name trainer hf python script/run_training.py yaml/whisper-params.yaml
```

or 

```bash
sudo docker run --gpus all -v ~/cache:/root/.cache/ --env-file .env -it --rm --name trainer -d hf bash
```

or 

```bash
sudo docker run --gpus all -v ~/cache:/root/.cache/ --env-file .env -it --rm --name trainer hf python script/run_training_track_with_mlflow.py yaml/whisper-params.yaml

```

```bash
sudo docker exec -it trainer python script/run_training.py yaml/whisper-params.yaml
```


-- remove
does not appear to have a file named preprocessor_config.json. Checkout
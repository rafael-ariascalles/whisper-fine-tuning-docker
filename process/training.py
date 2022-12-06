import subprocess

gpu_info = subprocess.run(["nvidia-smi"])
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
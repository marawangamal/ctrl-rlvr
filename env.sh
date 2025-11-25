# ###############################
# # MODULE SETUP (tamia / mila)
# ###############################
# module purge
# module load cuda/12.1           # Matches PyTorch cu121 wheels
# module load python/3.11         # Matches the conda env python=3.11


# ###############################
# # CREATE & ACTIVATE VENV
# ###############################
# python3.11 -m venv .venv
# source .venv/bin/activate

# # Always good practice:
# pip install --upgrade pip wheel setuptools


###############################
# PYTORCH + CUDA 12.1 (pip version)
# Equivalent of:
#   conda install pytorch torchvision torchaudio pytorch-cuda=12.1
###############################
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


###############################
# FAISS-GPU 1.8.0 (pip equivalent)
# Equivalent of:
#   conda install -c pytorch -c nvidia faiss-gpu=1.8.0
# Only needed for HMM distillation
###############################
pip install faiss-gpu==1.8.0.post1 --index-url https://download.pytorch.org/whl/cu121


###############################
# EXTRA DEPS FROM CTRL-G README
###############################
pip install \
  "transformers==4.41.2" \
  "huggingface_hub==0.23.4" \
  sentencepiece \
  protobuf \
  notebook \
  ipywidgets


###############################
# INSTALL CTRL-G LOCALLY (editable)
###############################
pip install git+https://github.com/joshuacnf/Ctrl-G


pip install \
  trl \
  datasets \
  tqdm \
  numpy \
  matplotlib \
  peft \
  wandb \
  math_verify \
  tiktoken 
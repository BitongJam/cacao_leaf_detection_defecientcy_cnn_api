
# 📜GUIDE FOR VERSION FOR INSTALLING
```
Library	Compatible Python Versions	Notes
TensorFlow 2.16+	Python 3.8–3.11	Ang TensorFlow 2.16 (latest stable release) wala pa support sa Python 3.12.
OpenCV (cv2)	Python 3.6–3.12	Compatible sa halos tanan recent versions.
PyTorch	Python 3.8–3.11	Same range as TensorFlow.
Keras	Dependent sa TensorFlow	Automatically works if TensorFlow works.
scikit-learn, NumPy, Pandas, Matplotlib	Python 3.8–3.12	No problem with 3.10 or 3.11.



Library             Stable Version (as of 2025)             Notes

Python              3.10.x                                  Most compatible
TensorFlow          2.16+                                   Works with 3.8–3.11
PyTorch             2.3+                                    Compatible with CUDA 12+
OpenCV              4.10+                                   Works fine
scikit-learn        1.5+                                    Stable
NumPy               1.26+                                   Optimized for ML workloads
```
# 📜Installation ng to Python-3.10.14 in ubuntu 24 Lts 

 ```
 sudo apt update
 ```
 ```
 sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
 ```
 ```
 cd /tmp
 ```
 ```
 wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
 ```
 ```
 tar -xf Python-3.10.14.tgz
 ```
 ```
 cd Python-3.10.14
 ```
 ```
 ./configure --enable-optimizations
 ```
 ```
 sudo make install
 ```
 ```
 python3 --version
 ```

# 📜Install script 

 ```
 sudo apt install python3 python3-pip python3-venv -y
 python3.10 -m venv env_lib-env
 source env_lib-env/bin/activate
 pip3 install wheel
 pip3 install -r requirements.txt
 deactivate
 exit
```
If the requirements do not exist yet, please install them first
 
```
for ubuntu 24 lts
===================
 pip install --upgrade pip
 pip install tensorflow # for ubuntu 24 effective
 pip install tflite-runtime # for pios effective
 pip install matplotlib
 pip install tensorflow-cpu
 sudo apt install cloud-init
 pip install python-multipart
 pip install fastapi uvicorn


 ```
 ```
 For raspberry Pi Os (legacy,64-bit) full
 ========================================
 #!/bin/bash

# ----------------------------------------
# Setup script for cacao_project
# Compatible with Python 3.11 and TensorFlow 2.12
# ----------------------------------------

# 1. Create virtual environment
python3 -m venv env_lib-env

# 2. Activate virtual environment
source env_lib-env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install compatible packages
pip install tensorflow==2.21.0 keras==3.13.2 numpy==2.4.3 pillow==12.1.1 fastapi uvicorn

# 5. Confirm installation versions
echo "Installed versions:"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import keras; print('Keras:', keras.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
python -c "import PIL; print('Pillow:', PIL.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import uvicorn; print('Uvicorn:', uvicorn.__version__)"

echo "Setup complete! Your environment is ready."
 ```
 # pag mag freeze kag library
 ```
 pip freeze > requirements.txt
 ```

# 📜Make gitignore
```
touch .gitignore
```
 add this to .gitignore 
 
 Python virtual environments
```
env/
venv/
env_lib-env/

# Compiled Python files
__pycache__/
*.pyc

# Large model/data files
*.h5
*.ckpt
*.pt
*.pkl
```


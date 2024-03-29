[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/FADS/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhanxiao0607%2FFADScr&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# FADScr: Few-shot Anomaly Detection and Classification Through Reinforced Data Selection with a Combinatorial Reward
A Pytorch implementation of [FADScr]().

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.9
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Instructions

Please download the datasets from Google Drive: https://drive.google.com/drive/folders/1B3Y8oIvr4bS4IBO-3YzyXw1W6uWfZAxh?usp=sharing

Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/FADScr.git my-project
        cd my-project

Run and test:

        python3 main_CERT.py
        or
        python3 main_IDS.py
        or
        python3 main_UNSW.py

## Citation

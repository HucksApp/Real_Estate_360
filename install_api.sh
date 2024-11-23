#!/bin/bash
sudo apt-get -y update &&
sudo apt-get -y upgrade;
sudo apt install python3.8-venv
wd=$(pwd)
cd Api;
python3 -m venv "$wd/Api/re_360" &&
source "$wd/Api/re_360/bin/activate" &&
pip3 install opencv-python
pip3 install numpy --upgrade

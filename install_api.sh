#!/bin/bash
wd=$(pwd)
cd Api;
python3 -m venv "$wd/re_360" &&
source "$wd/re_360" &&
pip3 install numpy opencv-python

#!/bin/bash

echo "开始下载Anaconda安装脚本"
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2023.03-Linux-x86_64.sh -O ~/anaconda_installer.sh

echo "安装Anaconda到用户目录下的anaconda3文件夹"
bash ~/anaconda_installer.sh -b -p $HOME/anaconda3

echo "删除安装脚本"
rm ~/anaconda_installer.sh

echo "将Anaconda添加到环境变量"
export PATH="$HOME/anaconda3/bin:$PATH"

echo "初始化conda"
conda init bash

echo "激活conda环境"
source ~/.bashrc

echo "安装软件包：python 3.9, pytorch, torchvision, torchaudio, pytorch-cuda 11.8"
conda install --yes python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

echo "安装软件包：jinja2 3.0.3, nbconvert 6.4.4, matplotlib"
conda install --yes jinja2=3.0.3 nbconvert=6.4.4 matplotlib

echo "安装完成！"
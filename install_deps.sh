#!/bin/sh 

echo "Installing pip3...................."
sudo apt-get update
sudo apt-get -y install python3-pip

echo "pip3 version......................."
pip3 --version

echo "Installing yaml and pandas library............"
pip3 install pyyaml
pip3 install --upgrade pandas

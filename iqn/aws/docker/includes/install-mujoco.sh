cd /
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev python3-numpy python3-scipy
pip3 install -r requirements.txt
python3 setup.py install
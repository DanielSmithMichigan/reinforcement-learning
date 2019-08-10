FROM ubuntu:18.04
COPY includes/ /included-scripts
RUN apt-get update
RUN apt-get install -y python3.5 python3-pip git libmysqlclient-dev vim
RUN pip3 install tensorflow matplotlib numpy mysqlclient mysql-connector
RUN bash /included-scripts/install-mujoco.sh
RUN bash /included-scripts/install-gym.sh
RUN bash /included-scripts/get-dqn.sh

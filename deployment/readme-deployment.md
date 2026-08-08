sudo ip addr add <HOST_IP>/<PREFIX_LENGTH> dev <NETWORK_INTERFACE>
sudo ip link set <NETWORK_INTERFACE> up
ping <ROBOT_IP>


python -m real_deployment.model_controller_del_ee


## install frankx
conda install frankx
unzip frankx.zip

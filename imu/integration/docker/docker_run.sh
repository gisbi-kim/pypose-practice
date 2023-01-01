# xhost +local:root

docker run --rm -it \
    --user root \
    --name 'pypose_imu' \
    --net=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env="DISPLAY" \
    --volume=/home/gskim/Documents/practices/pypose_imu_kinematics:/project \
    --volume=/home/gskim/Documents/data/kitti:/data \
    pypose:imu \
    /bin/bash -c 'cd /project/python; python3 main.py; bash'

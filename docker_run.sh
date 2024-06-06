docker run --rm --gpus=all --net=host --privileged -e=DISPLAY -e=XDG_RUNTIME_DIR --shm-size=2gb \
    -v `pwd`:${HOME}/repos/ERAS \
    -v ${HOME}/.ssh:${HOME}/.ssh \
    -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR \
    -v ${XAUTHORITY}:${XAUTHORITY} \
    -v /usr/local/share/ca-certificates:/usr/local/share/ca-certificates \
    -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/ca-certificates.crt \
    -w ${HOME}/repos/ERAS \
    -it $(docker build -q --build-arg uid=$(id -u ${USER}) --build-arg user=${USER} -t "local/robot_learning/er:latest" .)
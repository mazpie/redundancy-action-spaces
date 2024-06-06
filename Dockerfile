FROM nvidia/cuda:11.2.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# controls which driver libraries/binaries will be mounted inside the container
# docs: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
# This can be specifed at runtime or at image build time. Here we do it at build time.
# Graphics capability is required to be specified for rendering.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

ARG uid
ARG user

RUN \
  apt-get update && \
  apt-get install -y \
    sudo \
    python3-pip \
    git \
    zsh \
    curl \
    wget \
    unzip \
    tmux \
    vim \
    mesa-utils \
    xvfb \
    qtbase5-dev \
    qtdeclarative5-dev \
    libqt5webkit5-dev \
    libsqlite3-dev \
    qt5-default  \
    qttools5-dev-tools

RUN \
  useradd -u ${uid} ${user} && \
  echo "${user} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${user} && \
  chmod 0440 /etc/sudoers.d/${user} && \
  mkdir -p /home/${user} && \
  chown -R ${user}:${user} /home/${user} && \
  chown ${user}:${user} /usr/local/bin && \
  mkdir /tmp/.X11-unix && \
  chmod 1777 /tmp/.X11-unix && \
  chown root /tmp/.X11-unix

USER ${user}

WORKDIR /home/${user}

WORKDIR /home/${user}


RUN \
  cur=`pwd` && \
  wget http://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
  tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
  export COPPELIASIM_ROOT="$cur/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" && \
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/platforms && \
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT && \
  git clone https://github.com/stepjam/PyRep.git && \
  cd PyRep && \
  pip3 install -r requirements.txt && \
  pip3 install setuptools && \
  pip3 install .

RUN \
  git clone https://github.com/stepjam/RLBench.git && cd RLBench && \
  pip install -r requirements.txt && \
  pip install .

 RUN \
   mkdir -p ~/.config/fish

RUN \
  export cur=`pwd` && echo "set -x COPPELIASIM_ROOT $cur/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.config/fish/config.fish && \
  echo 'set -x LD_LIBRARY_PATH $LD_LIBRARY_PATH $COPPELIASIM_ROOT $COPPELIASIM_ROOT/platforms' >> ~/.config/fish/config.fish && \
  echo 'set -x QT_QPA_PLATFORM_PLUGIN_PATH $COPPELIASIM_ROOT' >> ~/.config/fish/config.fish

RUN sudo apt-get install -y fish

RUN pip3 install torch==2.0.1 hydra-core scipy shapely trimesh pyrender wandb==0.15.4 timm

# install VS Code (code-server)
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension ms-python.python ms-toolsai.jupyter

# install VS Code extensions
RUN sudo apt-get install wget gpg
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
RUN sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
RUN sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
RUN rm -f packages.microsoft.gpg
RUN sudo apt-get update
RUN sudo apt-get install -y code
# RUN code --install-extension ms-python.python ms-toolsai.jupyter

# Additional packages
RUN sudo apt-get install libglew2.1 libgl1-mesa-glx libosmesa6
RUN pip3 install gym termcolor hydra-submitit-launcher PyOpenGL==3.1.4 PyOpenGL_accelerate notebook matplotlib
RUN pip3 install --upgrade requests

COPY libvvcl.so CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
RUN pip3 install tensorboard imageio[ffmpeg] hydra-joblib-launcher moviepy

RUN echo 'set -x PATH $PATH $HOME/.local/bin' >> ~/.config/fish/config.fish

RUN \
  sudo apt-get update && sudo apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  x11-xserver-utils xvfb \
  && sudo apt-get clean

RUN pip3 install einops dm_env
  
ENTRYPOINT ["/usr/bin/fish"]

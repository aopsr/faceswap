# IMPORTANT: Follow the instructions below. Then, replace the installed files with those from this repo, including .git if you want to get updates.

# Easy install
- ## Windows 10

  Use one-click installer here: [faceswap_setup_x64.exe](https://github.com/deepfakes/faceswap/releases/download/v2.0.0/faceswap_setup_x64.exe), then replace the installed files with this repo.

- ## Linux
  Most Ubuntu/Debian or CentOS based Linux distributions will work.

  Download [faceswap_setup_x64.sh](https://github.com/deepfakes/faceswap/releases/download/v2.0.0/faceswap_setup_x64.sh) and run:

  `bash ./faceswap_setup_x64.sh`

  Replace the installed files with this repo.

- **macOS**
  Experimental support for GPU-accelerated, native Apple Silicon processing (e.g. Apple M1 chips). Installation instructions can be found [further down this page](#macos-apple-silicon-install-guide).
  Intel based macOS systems should work, but you will need to follow the [Manual Install](#manual-install) instructions.
- All operating systems must be 64-bit for Tensorflow to run.

Alternatively, there is a docker image that is based on Debian.


# Manual Install

## Prerequisites

### Anaconda
Download and install the latest Python 3 Anaconda from: https://www.anaconda.com/download/. Unless you know what you are doing, you can leave all the options at default.

### Git
Download and install Git for Windows: https://git-scm.com/download/win. Unless you know what you are doing, you can leave all the options at default.

## Setup
Reboot your PC, so that everything you have just installed gets registered.

### Anaconda
#### Set up a virtual environment
- Open up Anaconda Navigator
- Select "Environments" on the left hand side
- Select "Create" at the bottom
- In the pop up:
    - Give it the name: faceswap
    - **IMPORTANT**: Select python version 3.8
    - Hit "Create" (NB: This may take a while as it will need to download Python)
![Anaconda virtual env setup](https://i.imgur.com/CLIDDfa.png)

#### Entering your virtual environment
To enter the virtual environment:
- Open up Anaconda Navigator
- Select "Environments" on the left hand side
- Hit the ">" arrow next to your faceswap environment and select "Open Terminal"
![Anaconda enter virtual env](https://i.imgur.com/rKSq2Pd.png)

### faceswap
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Get the faceswap repo by typing: `git clone --depth 1 https://github.com/aopsr/faceswap.git`
- Enter the faceswap folder: `cd faceswap`

#### Easy install
- Enter the command `python setup.py` and follow the prompts:
- If you have issues/errors follow the Manual install steps below.

#### Manual install
Do not follow these steps if the Easy Install above completed succesfully.
If you are using an Nvidia card make sure you have the correct versions of Cuda/cuDNN installed for the required version of Tensorflow
- Install tkinter (required for the GUI) by typing: `conda install tk`
- Install requirements:
  - For Nvidia GPU users: `pip install -r ./requirements/requirements_nvidia.txt`
  - For AMD GPU users: `pip install -r ./requirements/requirements_amd.txt`
  - For CPU users: `pip install -r ./requirements/requirements_cpu.txt`

## Running faceswap
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Enter the faceswap folder: `cd faceswap`
- Enter the following to see the list of commands: `python faceswap.py -h` or enter `python faceswap.py gui` to launch the GUI

## Create a desktop shortcut
A desktop shortcut can be added to easily launch straight into the faceswap GUI:

- Open Notepad
- Paste the following:
```
%USERPROFILE%\Anaconda3\envs\faceswap\python.exe %USERPROFILE%/faceswap/faceswap.py gui
```
- Save the file to your desktop as "faceswap.bat"

## Updating faceswap
It's good to keep faceswap up to date as new features are added and bugs are fixed. To do so:
- If using the GUI you can go to the Help menu and select "Check for Updates...". If updates are available go to the Help menu and select "Update Faceswap". Restart Faceswap to complete the update.
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Enter the faceswap folder: `cd faceswap`
- Enter the following `git pull --all`
- Once the latest version has downloaded, make sure your dependencies are up to date. There is a script to help with this: `python update_deps.py`

# macOS (Apple Silicon) Install Guide

## Prerequisites

### OS
macOS 12.0+

### XCode Tools
```sh
xcode-select --install
```

### XQuartz
Download and install from:
- https://www.xquartz.org/

### Conda
Download and install the latest Conda env from:
- https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

Install Conda:
```sh
$ chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
$ sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
$ source ~/miniforge3/bin/activate
```
## Setup
### Create and Activate the Environment
```sh
$ conda create --name faceswap python=3.9
$ conda activate faceswap
```

### faceswap
- Download the faceswap repo and enter the faceswap folder:
```sh
$ git clone --depth 1 https://github.com/aopsr/faceswap.git
$ cd faceswap
```

#### Easy install
```sh
$ python setup.py
```

- If you have issues/errors follow the Manual install steps below.


# General Install Guide

## Installing dependencies
### Git
Git is required for obtaining the code and keeping your codebase up to date.
Obtain git for your distribution from the [git website](https://git-scm.com/downloads).

### Python
The recommended install method is to use a Conda3 Environment as this will handle the installation of Nvidia's CUDA and cuDNN straight into your Conda Environment. This is by far the easiest and most reliable way to setup the project.
  - MiniConda3 is recommended: [MiniConda3](https://docs.conda.io/en/latest/miniconda.html)

Alternatively you can install Python (>= 3.7-3.9 64-bit) for your distribution (links below.) If you go down this route and are using an Nvidia GPU you should install CUDA (https://developer.nvidia.com/cuda-zone) and cuDNN (https://developer.nvidia.com/cudnn). for your system. If you do not plan to build Tensorflow yourself, make sure you install the correct Cuda and cuDNN package for the currently installed version of Tensorflow (Current release: Tensorflow 2.9. Release v1.0: Tensorflow 1.15). You can check for the compatible versions here: (https://www.tensorflow.org/install/source#gpu).
  - Python distributions:
    - apt/yum install python3 (Linux)
    - [Installer](https://www.python.org/downloads/release/python-368/) (Windows)
    - [brew](https://brew.sh/) install python3 (macOS)

### Virtual Environment
  It is highly recommended that you setup faceswap inside a virtual environment. In fact we will not generally support installations that are not within a virtual environment as troubleshooting package conflicts can be next to impossible.

  If using Conda3 then setting up virtual environments is relatively straight forward. More information can be found at [Conda Docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

  If using a default Python distribution then [virtualenv](https://github.com/pypa/virtualenv) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io) may help when you are not using docker.


## Getting the faceswap code
It is recommended to clone the repo with git instead of downloading the code from http://github.com/deepfakes/faceswap and extracting it as this will make it far easier to get the latest code (which can be done from the GUI). To clone a repo you can either use the Git GUI for your distribution or open up a command prompt, enter the folder where you want to store faceswap and enter:
```bash
git clone https://github.com/aopsr/faceswap.git
```


## Setup
Enter your virtual environment and then enter the folder that faceswap has been downloaded to and run:
```bash
python setup.py
```
If setup fails for any reason you can still manually install the packages listed within the files in the requirements folder.

### About some of the options
   - CUDA: For acceleration. Requires a good nVidia Graphics Card (which supports CUDA inside)
   - Docker: Provide a ready-made image. Hide trivial details. Get you straight to the project.
   - nVidia-Docker: Access to the nVidia GPU on host machine from inside container.

# Docker Install Guide

## Docker General
<details>
  <summary>Click to expand!</summary>

  ### CUDA with Docker in 20 minutes.
  
  1. Install Docker
     https://www.docker.com/community-edition

  2. Install Nvidia-Docker & Restart Docker Service
     https://github.com/NVIDIA/nvidia-docker

  3. Build Docker Image For faceswap
  
  ```bash
  docker build -t deepfakes-gpu -f Dockerfile.gpu . 
  ```

  4. Mount faceswap volume and Run it
    a). without `gui.tools.py` gui not working.

 ```bash
 nvidia-docker run --rm -it -p 8888:8888 \
     --hostname faceswap-gpu --name faceswap-gpu \
     -v /opt/faceswap:/srv \
     deepfakes-gpu
 ```

    b). with gui. tools.py gui working.

Enable local access to X11 server

```bash
xhost +local:
```

Enable nvidia device if working under bumblebee

```bash
echo ON > /proc/acpi/bbswitch
```

Create container
```bash
nvidia-docker run -p 8888:8888 \
   --hostname faceswap-gpu --name faceswap-gpu \
   -v /opt/faceswap:/srv \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -e DISPLAY=unix$DISPLAY \
   -e AUDIO_GID=`getent group audio | cut -d: -f3` \
   -e VIDEO_GID=`getent group video | cut -d: -f3` \
   -e GID=`id -g` \
   -e UID=`id -u` \
   deepfakes-gpu

```

Open a new terminal to interact with the project

```bash
docker exec -it deepfakes-gpu /bin/bash
```

Launch deepfakes gui (Answer 3 for NVIDIA at the prompt)

```bash
python3.8 /srv/faceswap.py gui
```
</details>

## CUDA with Docker on Arch Linux

<details>
  <summary>Click to expand!</summary>

### Install docker

```bash
sudo pacman -S docker
```

The steps are same but Arch linux doesn't use nvidia-docker

create container

```bash
docker run -p 8888:8888 --gpus all --privileged -v /dev:/dev \  
            --hostname faceswap-gpu --name faceswap-gpu \
            -v /mnt/hdd2/faceswap:/srv \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -e DISPLAY=unix$DISPLAY \
            -e AUDIO_GID=`getent group audio | cut -d: -f3` \
            -e VIDEO_GID=`getent group video | cut -d: -f3` \
            -e GID=`id -g` \
            -e UID=`id -u` \
            deepfakes-gpu
```

Open a new terminal to interact with the project

```bash
docker exec -it deepfakes-gpu /bin/bash
```

Launch deepfakes gui (Answer 3 for NVIDIA at the prompt)

**With `gui.tools.py` gui working.**
 Enable local access to X11 server

 ```bash
xhost +local:
```
 
 ```bash
 python3.8 /srv/faceswap.py gui
 ```

</details>

--- 
## A successful setup log, without docker.
```
INFO    The tool provides tips for installation
        and installs required python packages
INFO    Setup in Linux 4.14.39-1-MANJARO
INFO    Installed Python: 3.7.5 64bit
INFO    Installed PIP: 10.0.1
Enable  Docker? [Y/n] n
INFO    Docker Disabled
Enable  CUDA? [Y/n]
INFO    CUDA Enabled
INFO    CUDA version: 9.1
INFO    cuDNN version: 7
WARNING Tensorflow has no official prebuild for CUDA 9.1 currently.
        To continue, You have to build your own tensorflow-gpu.
        Help: https://www.tensorflow.org/install/install_sources
Are System Dependencies met? [y/N] y
INFO    Installing Missing Python Packages...
INFO    Installing tensorflow-gpu
......
INFO    Installing tqdm
INFO    Installing matplotlib
INFO    All python3 dependencies are met.
        You are good to go.
```

## Run the project
Once all these requirements are installed, you can attempt to run the faceswap tools. Use the `-h` or `--help` options for a list of options.

```bash
python faceswap.py -h
```

or run with `gui` to launch the GUI
```bash
python faceswap.py gui
```


Proceed to [../blob/master/USAGE.md](USAGE.md)

## Notes
This guide is far from complete. Functionality may change over time, and new dependencies are added and removed as time goes on.

If you are experiencing issues, please raise them in the [faceswap Forum](https://faceswap.dev/forum) instead of the main repo. Usage questions raised in the issues within this repo are liable to be closed without response.

# Faceswap
<p align="center">
  <a href="https://faceswap.dev"><img src="https://i.imgur.com/zHvjHnb.png"></img></a>
<br />FaceSwap is a tool that utilizes deep learning to recognize and swap faces in pictures and videos.
</p>
<p align="center">
<img src = "https://i.imgur.com/nWHFLDf.jpg"></img>
</p>

<p align="center">
<a href="https://discord.gg/FC54sYg"><img src="https://i.imgur.com/gIpztkv.png"></img></a>
</p>

![Build Status](https://github.com/deepfakes/faceswap/actions/workflows/pytest.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/faceswap/badge/?version=latest)](https://faceswap.readthedocs.io/en/latest/?badge=latest)

Make sure you check out [INSTALL.md](INSTALL.md) before getting started.

- [faceswap](#faceswap)
- [How To setup and run the project](#how-to-setup-and-run-the-project)
- [Overview](#overview)
  - [Extract](#extract)
  - [Train](#train)
  - [Convert](#convert)
  - [GUI](#gui)
- [General notes:](#general-notes)
- [Help I need support!](#help-i-need-support)
  - [Discord Server](#discord-server)
  - [FaceSwap Forum](#faceswap-forum)
- [How to contribute](#how-to-contribute)
  - [For people interested in the generative models](#for-people-interested-in-the-generative-models)
  - [For devs](#for-devs)
  - [For non-dev advanced users](#for-non-dev-advanced-users)
  - [For end-users](#for-end-users)

# How To setup and run the project
FaceSwap is a Python program that will run on multiple Operating Systems including Windows, Linux, and MacOS.

See [INSTALL.md](INSTALL.md) for full installation instructions. You will need a modern GPU with CUDA support for best performance. AMD GPUs are partially supported.

# Overview
The project has multiple entry points. You will have to:
 - Gather photos and/or videos
 - **Extract** faces from your raw photos
 - **Train** a model on the faces extracted from the photos/videos
 - **Convert** your sources with the model

Check out [USAGE.md](USAGE.md) for more detailed instructions.

## Extract
From your setup folder, run `python faceswap.py extract`. This will take photos from `src` folder and extract faces into `extract` folder.

## Train
From your setup folder, run `python faceswap.py train`. This will take photos from two folders containing pictures of both faces and train a model that will be saved inside the `models` folder.

## Convert
From your setup folder, run `python faceswap.py convert`. This will take photos from `original` folder and apply new faces into `modified` folder.

## GUI
Alternatively, you can run the GUI by running `python faceswap.py gui`

# General notes:
- All of the scripts mentioned have `-h`/`--help` options with arguments that they will accept. You're smart, you can figure out how this works, right?!

NB: there is a conversion tool for video. This can be accessed by running `python tools.py effmpeg -h`. Alternatively, you can use [ffmpeg](https://www.ffmpeg.org) to convert video into photos, process images, and convert images back to the video.


**Some tips:**

Reusing existing models will train much faster than starting from nothing.
If there is not enough training data, start with someone who looks similar, then switch the data.

# Help I need support!
## Discord Server
Your best bet is to join the [FaceSwap Discord server](https://discord.gg/FC54sYg) where there are plenty of users willing to help. Please note that, like this repo, this is a SFW Server!

## FaceSwap Forum
Alternatively, you can post questions in the [FaceSwap Forum](https://faceswap.dev/forum). Please do not post general support questions in this repo as they are liable to be deleted without response.

# How to contribute

## For people interested in the generative models
 - Go to the 'faceswap-model' to discuss/suggest/commit alternatives to the current algorithm.

## For devs
 - Read this README entirely
 - Fork the repo
 - Play with it
 - Check issues with the 'dev' tag
 - For devs more interested in computer vision and openCV, look at issues with the 'opencv' tag. Also feel free to add your own alternatives/improvements

## For non-dev advanced users
 - Read this README entirely
 - Clone the repo
 - Play with it
 - Check issues with the 'advuser' tag
 - Also go to the '[faceswap Forum](https://faceswap.dev/forum)' and help others.

## For end-users
 - Get the code here and play with it if you can
 - You can also go to the [faceswap Forum](https://faceswap.dev/forum) and help or get help from others.
 - Be patient. This is a relatively new technology for developers as well. Much effort is already being put into making this program easy to use for the average user. It just takes time!
 - **Notice** Any issue related to running the code has to be opened in the [faceswap Forum](https://faceswap.dev/forum)!

# Implementation of Exploring Convolutional Neural Networks for Voice Activity Detection
Implementation of the VAD model created in the paper with the same name written by Diego Augusto Silva, José Augusto Stuchi, Ricardo P. Velloso Violato and Luís Gustavo D. Cuozzo

This is not an official repository of the authors, I merely created it as I wanted to use their model for my own project.

The official paper can be found here:
Silva, Diego Augusto, et al. "Exploring convolutional neural networks for voice activity detection." Cognitive Technologies. Springer, Cham, 2017. 37-47.

## Structure
The project contains different classes for the different parts. To run the whole program, execute the run.py file.

In order to get the program to work, path_data needs to be replaced with the path to the QUT-NOISE-TIMIT dataset.

The sound_viewer_tool is a copy of the tool originally created at https://github.com/ljvillanueva/Sound-Viewer-Tool/blob/master/svt.py.
As the tool was discontinued and parts were not working for me, I adapted their code.
The changes are, that audiolab has been replaced with librosa and PIL instead of Image is imported. Also, parts of the code have been adjusted to work with Python3.

## TODO
To fully copy the methodology of the paper, the following changes need to be done:
- change optimizer as well as the learning rate
- implement in data creation that each frame only has 1 label and frame sizes are updated accordingly. Currently, if speech is found in a frame, it is always considered speech no matter the percentage of speech found.

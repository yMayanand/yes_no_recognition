### This is Voice Recognition Project Implemented in python using Pytorch Framework.
- ### Dataset Used
[Yes No Dataset](https://www.openslr.org/1/) this dataset contains 60 **YES** and **NO** speech samples

- ### Model Used
The strategy to train a speech recognition Model was to first convert raw audio waveform intp **spectrogram** and then treating this task as image recognition task. The model consists of two components first one encoder and another is decoder
- Encoder: Encoder is an convolutional network which convert our image into useful compressed representation.
- Decoder: Decoder consists of LSTM network to decode speech from images encoded representation.

- ### Loss Functions
Loss function used here is **CTC** loss function as we dont have  alignment knowledge between speech waveform and labels.

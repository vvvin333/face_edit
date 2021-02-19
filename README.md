## Face_turns
Experiments with face turns along yaw and pitch axis with ## StyleGAN &mdash;
I reconstructed stylegan`s work from several repositories for my images, combine them, tried to [find pitch_boundary](https://colab.research.google.com/drive/1xBtH-c1hmhoZ6X8KIpxyYB1li3x38ipE?usp=sharing)(for pitch-axis) from several photos with my 'by-eye' scores. Added move_images_in_latent_space.py script. 

## Samples
...


## Requirements:
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.10](https://img.shields.io/badge/tensorflow-1.10-green.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
I used conda (channels: conda-forge pytorch) for the packages:<br>
tqdm <br>
numpy <br>
pillow <br>
tensorflow==1.15.0 <br>
tensorflow-gpu <br>
keras <br>
dlib <br>
opencv <br>
imutils <br>

## Instructions:
1) (optional) [download pre-trained generator model](https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ) and  [pre-trained perceptual model](https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2) to the models/ folder.
2) (optional) for search face and align (one can then choose good enough images):
```
python align_images.py raw_images/ aligned_images/
```
3) (optional) encode photo to latent vector:
```
python encode_images.py --optimizer=lbfgs --face_mask=True --iterations=10 --use_l1_penalty=0.2 --use_lpips_loss=0 --use_discriminator_loss=0
```
4) edit photo in latent space with boundary (axis=pitch/yaw): 
```
python move_images_in_latent_space.py --axis=yaw --number_interpolation_steps=15 --morph_strength=3
```

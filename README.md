# Speed Estimation & Collision Prevention

In recent years, deep neural networks have been used for many perception based applications, from smart phones to self-driving cars. Digital cameras are some of the most important sensor that autonomous vehicle relies. Research have shown that it's possible to map raw pixels from a single front-facing camera directly to steering commands. However, little work has been done regarding predicting the speed of a moving vehicle. 

Here I propose a novel architecture based on Carreira and Zisserman's i3D model \cite{i3d}. First, the network demonstrated the ability to utilize spatial-temporal information to estimate the speed of a moving vehicle through very low validation loss. Secondly, the model also performed very well with visualization techniques, showing that the network understands when to stop and yield for other vehicle. 

## How to Use

For using the pretrained models for inferencing, please following the instructions below. 

	from speed_predictor import SpeedPredictor
	predictor = SpeedPredictor(<MODEL_PATH>)
	speed = predictor.predict_speed(<INPUT_IMAGE>)
	
For more detailed documentations, please refer to the source code. 

## The i3D Model
Both the temporal and spacial information will be critical to regress the speed of a moving vehicle. Naturally, this task fits the problem description of a video sequence classification problem. Inception3D, initially proposed by DeepMind, is a novel, state-of-the-art architecture for visual action classification \cite{i3d}.

This model is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatial-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters \cite{i3d}.

The i3D paper re-evaluates best architectures in light of the new Kinetics Human Action Video dataset, which consists of 400 human action classes and over 400 clips per class. We provide an analysis on how current architectures fare on the task of action classification on this dataset and how much performance improves on the smaller benchmark datasets after pre-training on Kinetics. 

After slighting tweaking the final layers of the network by adding one flatten and one dense layer, the model is suited for the regression problem. Here are the training results for the i3D network \cite{i3d}, as well as some comparison with the traditional ConvNets.


## About me
I am a young and passionate software engineer, machine learning researcher, and robotics enthusiast. You can find out more about my work by simply visiting my [Github page](https://github.com/NeilNie) or the links below. Thanks! 

[LinkedIn](https://www.linkedin.com/in/yongyang-neil-nie/) | [Website](https://neilnie.com) | [TEDxTalk](https://www.youtube.com/watch?v=SN2BZswEWUA) | [App Store](https://itunes.apple.com/us/developer/yongyang-nie/id1341231595?mt=8)


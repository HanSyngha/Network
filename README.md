To use the included code, you will need tensorflow 2.3.0-gpu verson ( This match the cuda verson 10.2 )
If you use docker, you can pull the image by

	sudo nvidia-docker pull tensorflow/tensorflow:2.3.0-gpu

Then you can make docker container by 

	sudo nvidia-docker run -it --name [name for your docker container] -v /[folder which you want to attach to your container]:/ tensorflow/tensorflow:2.3.0-gpu /bin/bash

you will need to install bellow packages in the docker container to run the code

	apt-get update
	apt-get install pillow
	pip install keras
	pip install tensorflow_datasets
	pip install opencv-python
	
Inside the code, you can choose the gpu number to use by
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = '[GPU number]'

Keras uses all of one gpu memorys so when the gpu you choose is running by other process, the code will not be executed

In the datasets folder, Imagenet has the original Imagenet images,
	training_features.npy => middle features which is made with 80% of Imagenet training data
	training_label.npy => label of 1000 classes, 80% of Imagenet training data
	validation_features.npy => middle features which is made with 20% of Imagenet training data
	validation_label.npy => label of 1000 classes, 20% of Imagenet training data
	test_features.npy => middle features which is made with Imagenet validation data
	test_label.npy => label of 1000 classes, Imagenet validation data
Remember we splited the training datas to use as training/validation and validation data as test set
You can find the spliting process in VGG_training.py code 
also remeber that we defined input to pool5 layers of VGG16 as front layers and flatten, three FC layers as back layers

The Back_layers_training_codes are the codes to train the back layers by condition
	Feature_map_error_model.py is the code training the back layers with one channel error
	single_packet_error_model.py is the code training the back layers with one packet error
	double_packet_error_model.py is the code training the back layers with double continuous double packet error
	triple_packet_error_model.py is the code training the back layers with triple continuous triple packet error
The above models will make 10 models by error situation with different taining epoches condition

	total_feature_map_error_model.py is the code training the back layers with total channel error ( first image with channel 0 error, second image with channel 1 error, and so on )
	total_packet_error_model.py is the code training the back layers with total packet error ( first image with pakcet 0 error, second image with packet 1 error, and so on )
Remember we defined the packet size as general MSS size ( 1460 bytes ) so each packet will have 8 channel
The middle feature of one images is 7x7x512 so we have 512 channels and 64 packets in one image

The folders in Back_layers_training_code is the folder to save the trained model.

The Evaluate_codes has the code for each training situation
Evaluate_random_double_error_~ codes are for testing the all double packet error situations (0,1) packet error to (62,63) packet error

The original_models are the models which we have trained
	original_back_layers_model.h5 is the back layers of pretrained VGG16. You have to load it with "load_model" function
	original_back_layers_weights.h5 is the weights of back layers of pretrained VGG16. You have to load it with "load_weights" method
	origianl_vgg_model.h5 is the pretrained VGG16 with learning rate 0.01 SDG optimizer. Please test both "load_model" and "load_weights"




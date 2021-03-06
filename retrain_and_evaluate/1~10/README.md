@210201 - 210221
	model training which i done is not well trained(Acc flucuated, loss increasing as epoch increase)
	so, try to find well-suit parameter for training 1_packet_drop feature
	Then find it.
	well-trained model in 1_packet_error_casei
		pretrained model : train done in 2020, load VGG16(keras.application.VGG16) model weighted of Imagenet, optimizer : SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True), batch_size=256, epochs=84
		start with pretrained model, optimizer : Adam(lr=5e-5,decay=0.001, beta_1=0.9,beta_2=0.999,epsilon=None, amsgrad=Trus,epoch=15) 
		Acc : 65.87%(compared with 65.89%,65.62%. Each corresponding with evaluating 0 packet loss of feature, 1 packet loss of that in pretrained model above
@210222 - 210302
First,
	try to verify whether	training $i_packet_drop model can recover corresponding packet_error_feature,
	it only applied to packet_drop 1 model(training again with 1_packet_drop feature)
@210304 - 
Then,
	3 thing i do
	it is concerned with after-pooling5 feature
	back_layer = fc1,fc2,fc3	
	well-trained model in 1_packet_error case : described in 210201~210221

	1. train back_layer(3 layer) weight-initial model with errored feature(1~25 packet error featured)
		-> 210306 about 08:20 python init_train.py 1~10(GPU 2)
	2. pick well-trained model in 1_packet_error case, evaluate 2~32 packet loss cases
		-> exec 210306 about 20:00pm python tmp/make_best_acc_model_in_packetdrop_1.py (GPU 3)
		 
	3. Student who is in the Master' course advised that the dataset used for train is not well-preprocessed, such as rescale (he knows that is not used in pretrained VGG16)
		-> extracted feature of pooling5 from not rescale dataset
		-> train with 6 packet error feature / use Adam(lr=5e-5,beta_1=0.9,beta_2=0.999,epsilon=None, decay=1e-6,amsgrad=False)  Categorical_accuracy
		-> test dataset
		28 th result == loss :2.37, Acc : 53.46% (loss min point)
		95 th result == loss 2.992, ACC : 55.49% (Acc max point)
		However, val_loss is 46.0698(28th epoch) -> 132.3214(95th epoch) : abnormal situation
		210306 idea
		# try to find optimal preprocessing condition, then train real model
		# use no-scale feature, train VGG16(weigth='imagenet'), not pretrained by predecessor

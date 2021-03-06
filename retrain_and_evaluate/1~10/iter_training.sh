#!/bin/sh

for i in $(seq 2 10)
do
	if [ "$i" -eq 6 ]; then
		#echo "is_"$i
		knockknock telegram --token 1531143270:AAF4Yz0AqL5hqexw16hwVVsTxwMMnSGVvQ0 --chat-id 1597147353 python train_1_layer.py $i 2>&1 | tee output/output_1_layer_Adam_drop_$i".txt"
		#continue
	else
		echo $i

#		knockknock telegram --token 1531143270:AAF4Yz0AqL5hqexw16hwVVsTxwMMnSGVvQ0 --chat-id 1597147353 python packet_drop.py $i 2>&1 | tee result/mk_output_feature_drop_$i".txt" &  
		knockknock telegram --token 1531143270:AAF4Yz0AqL5hqexw16hwVVsTxwMMnSGVvQ0 --chat-id 1597147353 python train_1_layer.py $i 2>&1 | tee output/output_1_layer_Adam_drop_$i".txt"
	fi
	
done

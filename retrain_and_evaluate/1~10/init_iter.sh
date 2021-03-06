#!/bin/sh

for i in 1 10 $(seq 2 9)
do
	echo "init training "$i
	knockknock telegram     --token 1531143270:AAF4Yz0AqL5hqexw16hwVVsTxwMMnSGVvQ0     --chat-id 1597147353     python init_train.py $i 2>&1 | tee output/output_init_train.txt
	
done

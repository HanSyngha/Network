#!/bin/sh
knockknock telegram \
    --token 1531143270:AAF4Yz0AqL5hqexw16hwVVsTxwMMnSGVvQ0 \
    --chat-id 1597147353 \
    python packet_drop.py 1 2>&1 | tee output/output_packet_drop_1.txt
knockknock telegram \
    --token 1531143270:AAF4Yz0AqL5hqexw16hwVVsTxwMMnSGVvQ0 \
    --chat-id 1597147353 \
    python tmp/make_best_acc_model_in_packetdrop_1.py 1 2>&1 | tee output/output_make_best_acc_model_in_packetdrop_1.txt



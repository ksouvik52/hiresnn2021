Version on which the models were tested:
========================================
Pytorch version: 1.5.1
Python version: 3.8.3

Run instructions:
==================
1. HIRE SNN testing:
====================
a) To test HIRE SNN: select folder/file location: HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_model.pt (as pretrained_snn)
and 
HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_bb_test_model.pt as pretrained_snn_bb(this is for black box testing only)
and edit in the run_snn_hire_test.py file.  
1. b) run command: python run_snn_hire_test.py

2. Traditional SNN testing:
===========================
a) To test traditional SNN: select folder/file location: traditional_models/vgg11_cifar100_tradit_model.pt as pretrained_snn
and 
traditional_models/vgg11_cifar100_tradit_bb_test_model.pt as pretrained_snn_bb (this is for black box testing only)
2. b) run command: python run_snn_tradit_test.py




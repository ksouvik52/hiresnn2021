<p align="center"><img width="30%" src="/Fig/Capture.PNG"></p><br/> 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


Welcome to the official repo of the `ICCV 2021` paper **`HIRE-SNN: Harnessing the Inherent Robustness of Energy-Efficient Deep Spiking Neural Networks by Training with Crafted Input Noise`**.

**This repo currently contains the test codes. Training code will be updated soon!**

### Authors:
1. **Souvik Kundu** (souvikku@usc.edu)
2. Massoud Pedram (pedram@usc.edu)
3. Peter A. Beerel (pabeerel@usc.edu)

### Abstract:
Low-latency deep spiking neural networks (SNNs) havebecome  a  promising  alternative  to  conventional  artificial neural networks (ANNs) because of their potential for increased  energy  efficiency  on  event-driven  neuromorphic hardware.Neural  networks,  including  SNNs,  however,  are subject to various adversarial attacks and must be trained to  remain  resilient  against  such  attacks  for  many  applications.   Nevertheless,  due  to  prohibitively  high  training costs associated with SNNs,  an analysis and optimization of deep SNNs under various adversarial attacks have beenlargely  overlooked.   In  this  paper,  we  first  present  a  detailed  analysis  of  the  inherent  robustness  of  low-latency SNNs against popular gradient-based attacks, namely fast gradient sign method (FGSM) and projected gradient descent  (PGD).  Motivated  by  this  analysis,  to  harness  themodel’s robustness against these attacks we present an SNN training  algorithm  that  uses  crafted  input  noise  and  incurs  no  additional  training  time.    To  evaluate  the  merits  of  our  algorithm,  we  conducted  extensive  experiments with variants of VGG and ResNet on both CIFAR-10 and CIFAR-100 dataset.  Compared to standard trained direct-input SNNs, our trained models yield improved classification  accuracy  of  up  to 13.7% and 10.1% on  FGSM  and PGD  attack  generated  images,  respectively,  with  negligible  loss  in  clean  image  accuracy.   Our  models  also  outperform inherently-robust SNNs trained on rate-coded in-puts  with  improved  or  similar  classification  performanceon  attack-generated  images  while  having  up  to 25x and ∼4.6x lower latency and computation energy, respectively.

<p align="center"><img width="45%" src="/Fig/intro_vgg11_sa_and_attack_performance_plot.png" /><img width="30.45%"  src="/Fig/Training_procedure.png" /></p><br/> 

### Version on which the models were tested:

* PyTorch version: `1.5.1`.
* Python version: `3.8.3`.

### Model download:
#### A. HIRE SNN models:
1. [vgg11_cifar100_hiresnn_tstep8_bb_test_model](https://drive.google.com/file/d/1cjNVZ0wx6R8cVJD9ivU9GBOL0wr6ZCeu/view?usp=sharing)
2. [vgg11_cifar100_hiresnn_tstep8_model](https://drive.google.com/file/d/1MdOyBL_NqMxgsTy_c62Xc7-ek2pO2DPB/view?usp=sharing)
#### B. Traditional SNN models:
1. [vgg11_cifar100_tradit_bb_test_model](https://drive.google.com/file/d/11NmAGUmbZ4WD3U1DMoQs5tDwbBDG-rjD/view?usp=sharing)
2. [vgg11_cifar100_tradit_model](https://drive.google.com/file/d/1GgW-dITrh2reHz6SadqpgN0RTGWoJGRm/view?usp=sharing)
### To test adversarial accuracy of a saved model, please follow these steps:
Create two folders named *`HIRE_SNN_models`* and *`traditional_models`*. Download the models to their respective folder locations.
#### 1. HIRE SNN testing:
a) To test HIRE SNN: select folder/file location: HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_model.pt (as pretrained_snn)
and 
HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_bb_test_model.pt as pretrained_snn_bb(this is for black box testing only)
and edit in the run_snn_hire_test.py file.  1. b) run command: python run_snn_hire_test.py

#### 2. Traditional SNN testing:
a) To test traditional SNN: select folder/file location: traditional_models/vgg11_cifar100_tradit_model.pt as pretrained_snn
and 
traditional_models/vgg11_cifar100_tradit_bb_test_model.pt as pretrained_snn_bb (this is for black box testing only)
2. b) run command: python run_snn_tradit_test.py

### Cite this work
If you find this project useful to you, please cite our work:

    @InProceedings{Kundu_2021_ICCV,
    author    = {Kundu, Souvik and Pedram, Massoud and Beerel, Peter A.},
    title     = {HIRE-SNN: Harnessing the Inherent Robustness of Energy-Efficient Deep Spiking Neural Networks by Training With Crafted Input Noise},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5209-5218}}

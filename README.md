# Neural_Dropout_Search

Code for the DAC'24 paper  *Hardware-Aware Neural Dropout Search for Reliable Uncertainty Prediction on FPGA*

The code main refer to the codebase (https://github.com/hmarkc/MCME_FPGA_Acc). 


## Software
There are 3 folders. In each folder, supernet_NMS.py implements training and search process. You can set corresponding flags to select experiments.

## Hardware
The hls4ml implementation for generating dropout-based Bayesian Neural Network.

## Citation
If you found it helpful, pls cite us using:


``` 
@inproceedings{zhang2024hardware,
  title={Hardware-Aware Neural Dropout Search for Reliable Uncertainty Prediction on FPGA},
  author={Zhang, Zehuan and Fan, Hongxiang and Chen, Hao and Dudziak, Lukasz and Luk, Wayne},
  booktitle={Proceedings of the 61st ACM/IEEE Design Automation Conference},
  pages={1--6},
  year={2024}
}

```

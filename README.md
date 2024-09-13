<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis </b></h2>
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{cheng2024convtimenet,
  title={Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis},
  author={Cheng, Mingyue and Yang, Jiqian and Pan, Tingyue and Liu, Qi and Li, Zhi},
  journal={arXiv preprint arXiv:2403.01493},
  year={2024}
}
```



## Project Overview

This is the official open source code for ConvTimenet.

paper link: [ConvTimeNet: A Deep Hierarchical Fully Convolutional Model for Multivariate Time Series Analysis](https://arxiv.org/abs/2403.01493) 

## About the ConvTimeNet

In this study, we delved into the research question of how to reinvigorate the role of convolutional networks in time series analysis modeling. 

![image](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/6ad67b14-ec3a-42b4-970f-585108a13bd6)


The ConvTimeNet is a deep hierarchical fully convolutional network, which can serve as a versatile backbone network for time series analysis. One key finding of ConvTimeNet is that preserving a deep and hierarchical convolutional network, equipped with modern techniques, can yield superior or competitive performance compared to the prevalent Transformer network and pioneering convolutional model. Extensive experiments conducted on the forecasting and classification of time series fully substantiate its effectiveness. Overall, we hope that ConvTimeNet can serve as an alternative model and encourage the research community to rethink the importance of convolution in time series mining tasks.

## Deformable Patch Embedding

![image](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/115bd0cd-c011-468e-b305-12526e773225)

 The deformable patch embedding plays a vital role in the performance of ConvTimeNet, masterfully tokenizing time series data through its adaptive adjustment of the patch size and position.


## ConvTimeNet Block

![ConTimeNet_backbone](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/5ee724c0-3956-492a-9601-82a235ed7ffc)

The ConvTimeNet Block has three key designs: (1) Different sizes of Convolution cores, to capture features on different time scales. (2) learnable residual, to make the network deeper. (3) Deep-wise convolution, compared with ordinary convolution, has less computation and improves the efficiency of the model.

## Main Results
Our ConvTimeNet achieves the effect of SOTA on time series classification task and time series long term prediction task.
![cd-diag](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/d1ef9c1a-2d0a-4c91-b02c-6390221868b3)

![radar](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/51cd735d-cee0-413f-8f49-d97e5334f367)

## Installration

1. Install requirements. pip install -r requirements.txt

2. Download data. You can download all the datasets from [AutoFormer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). For each csv file, create a folder ./dataset/{csv_name} and put the csv file into it.

3. Training. All the scripts are in the directory ./scripts. Please read the ./run_longExp.py to get the meanings of the parameters in the scripts.

## Citation

```
@article{cheng2024convtimenet, 
  title={Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis}, 
  author={Cheng, Mingyue and Yang, Jiqian and Pan, Tingyue and Liu, Qi and Li, Zhi}, 
  journal={arXiv preprint arXiv:2403.01493}, 
  year={2024} 
}
```

<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Convtimenet: A deep hierarchical fully convolutional model for multivariate\\ time series analysis (ACM WWW2025, Accepted) </b></h2>
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

### Further Reading

1, [**FormerTime: Hierarchical Multi-Scale Representations for Multivariate Time Series Classification**](https://arxiv.org/pdf/2302.09818).

**Authors**: Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Li, Zhi and Luo, Yucong and Chen, Enhong

```bibtex
@inproceedings{cheng2023formertime,
  title={Formertime: Hierarchical multi-scale representations for multivariate time series classification},
  author={Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Li, Zhi and Luo, Yucong and Chen, Enhong},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1437--1445},
  year={2023}
}
```

2, [**InstructTime: Advancing Time Series Classification with Multimodal Language Modeling**](https://arxiv.org/pdf/2403.12371).

**Authors**: Cheng, Mingyue and Chen, Yiheng and Liu, Qi and Liu, Zhiding and Luo, Yucong

```bibtex
@article{cheng2024advancing,
  title={Advancing Time Series Classification with Multimodal Language Modeling},
  author={Cheng, Mingyue and Chen, Yiheng and Liu, Qi and Liu, Zhiding and Luo, Yucong},
  journal={arXiv preprint arXiv:2403.12371},
  year={2024}
}
```

3, [**TimeMAE: Self-supervised Representation of Time Series with Decoupled Masked Autoencoders**](https://arxiv.org/pdf/2303.00320).

**Authors**: Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong

```bibtex
@article{cheng2023timemae,
  title={Timemae: Self-supervised representations of time series with decoupled masked autoencoders},
  author={Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong},
  journal={arXiv preprint arXiv:2303.00320},
  year={2023}
}
```

4, [**CrossTimeNet: Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model**](https://arxiv.org/pdf/2403.12372).

**Authors**: Cheng, Mingyue and Tao, Xiaoyu and Liu, Qi and Zhang, Hao and Chen, Yiheng and Lei, Chenyi

```bibtex
@article{cheng2024learning,
  title={Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model},
  author={Cheng, Mingyue and Tao, Xiaoyu and Liu, Qi and Zhang, Hao and Chen, Yiheng and Lei, Chenyi},
  journal={arXiv preprint arXiv:2403.12372},
  year={2024}
}
```







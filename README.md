Pytorch Implementation (v1) for  
# From Semantic Categories to Fixations: A Novel Weakly-supervised Visual-auditory Saliency Detection Approach,   
![net](https://github.com/CVPR2021Submit/STANet/blob/main/fig/net.gif)  
## Abstract
Thanks to the rapid advances in the deep learning techniques and the wide availability of large-scale training sets, the performances of video saliency detection models have been improving steadily and significantly. However, the deep learning based visual-audio fixation prediction is still in its infancy. At present, only a few visual-audio sequences have been furnished with real fixations being recorded in the real visual-audio environment. Hence, it would be neither efficiency nor necessary to re-collect real fixations under the same visual-audio circumstance. To address the problem, this paper advocate a novel approach in a weaklysupervised manner to alleviating the demand of large-scale training sets for visual-audio model training. By using the video category tags only, we propose the selective class activation mapping (SCAM), which follows a coarse-to-fine strategy to select the most discriminative regions in the spatial-temporal-audio circumstance. Moreover, these regions exhibit high consistency with the real human-eye fixations, which could subsequently be employed as the pseudo GTs to train a new spatial-temporal-audio (STA) network. Without resorting to any real fixation, the performance of our STA network is comparable to that of the fully supervised ones.  
## Dependencies
* GTX1080Ti
* python 3.6  
* Matlab R2016b  
* pytorch 1.2.0  
* soundfile  
* scipy  
## Preparation
1.Downloading the official pretrained model  
`net = torch.hub.load('facebookresearch/WSL-Images','resnext101_32x8d_wsl')` of ResNeXt implemented in Pytorch, and  
[vggsound](https://github.com/hche11/VGGSound), `net = torch.load('vggsound_netvlad')`, if you want to train/test the network.  
2.Downloading the [AVE](https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view) dataset(the training dataset), [AVAD](https://sites.google.com/site/minxiongkuo/home), [DIEM](https://thediemproject.wordpress.com/videos-and%c2%a0data/), [SumMe](https://gyglim.github.io/me/vsum/index.html#benchmark), [ETMD](http://cvsp.cs.ntua.gr/research/aveyetracking/), [Coutrot](http://antoinecoutrot.magix.net/public/databases.html) datasets(the testing dataset) in the folder of data for training or test.  
## Note 
We use Fourier-transformed audio features as input, therefore, we firstly need to use the function audiostft.py to convert the audio files (.wav) to get the audio features(.h5).
## Training
- Stage 1. training the model of S<sub>coarse</sub>, ST<sub>coarse</sub>, SA<sub>coarse</sub> respectively using original AVE frames.  
- Stage 2. testing the model of S<sub>coarse</sub>, ST<sub>coarse</sub>, SA<sub>coarse</sub> respectively using original AVE frames.  
- Stage 3. training the model of S<sub>fine</sub>, ST<sub>fine</sub>, SA<sub>fine</sub> respectively using the crop data of AVE.   
- Stage 4. testing the model of S<sub>fine</sub>, ST<sub>fine</sub>, SA<sub>fine</sub> using the crop data respectively and generate the pseudoGT of AVE dataset.   
- Stage 5. training the model of STANet using the original AVE frames with generated pseudoGT.  
- The current version of the code is very complicated, the current work will continue, and this code will also be maintained. In the next version, we will merge the relevant code and refine the code to make it easier to read.
## Testing 
Firstly you need to use the function audiostft.py to convert the audio files (.wav) to get the audio features(.h5).
After the preparation, run this commond  
`python test.py`  
We provide the trained model file ([Baidu Netdisk](https://pan.baidu.com/s/1nvtJm1Z6-sHBaLPsEHhw4Q), code:6afo).  
The saliency maps are also available ([Baidu Netdisk](https://pan.baidu.com/s/1nvtJm1Z6-sHBaLPsEHhw4Q), code:6afo).  
## Evaluation
We use the evaluation code in the paper of [STAVIS](https://github.com/atsiami/STAViS) for fair comparisons.   
You may need to revise the algorithms, data_root, and maps_root defined in the main.m.   
The saliency maps of the competitors(ITTI, GBVS, SCLI, AWS-D, SBF, WSS, MWS, WSSA) are provided ([Baidu Netdisk](https://pan.baidu.com/s/1nvtJm1Z6-sHBaLPsEHhw4Q), code:6afo).  
### Quantitative comparisons:  
![Quantitative](https://github.com/CVPR2021Submit/STANet/blob/main/fig/cvpr2021.gif)  
### Qualitative comparisons:  
![Quantitative](https://github.com/CVPR2021Submit/STANet/blob/main/fig/compare.gif)  
## References
[1][Tsiami, A., Koutras, P., Maragos, P.STAViS: Spatio-Temporal AudioVisual Saliency Network. (CVPR 2020).](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tsiami_STAViS_Spatio-Temporal_AudioVisual_Saliency_Network_CVPR_2020_paper.pdf)  
[2][Tian, Y., Shi, J., Li, B., Duan, Z., Xu, C. Audio-Visual Event Localization in Unconstrained Videos. (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf)  
[3][Chen, H., Xie, W., Vedaldi, A., & Zisserman, A. Vggsound: A Large-Scale Audio-Visual Dataset. (ICASSP 2020)](https://www.robots.ox.ac.uk/~vgg/publications/2020/Chen20/chen20.pdf)  

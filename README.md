# TYolov5: A Temporal Yolov5 Detector Based on Quasi-Recurrent Neural Networks for Real-Time Handgun Detection in Video

Timely handgun detection is a crucial problem to improve public safety; nevertheless, the effectiveness of many surveillance systems still depends of finite human attention. Much of the previous research on handgun detection is based on static image detectors, leaving aside valuable temporal information that could be used to improve object detection in videos. To improve the performance of surveillance systems, a real-time temporal handgun detection system should be built. Using Temporal Yolov5, an architecture based on Quasi-Recurrent Neural Networks, temporal information is extracted from video to improve the results of handgun detection. Moreover, two publicly available datasets are proposed, labeled with hands, guns, and phones. One containing 2199 static images to train static detectors, and another with 5960 frames of videos to train temporal modules. Additionally, we explore two temporal data augmentation techniques based on Mosaic and Mixup. The resulting systems are three temporal architectures: one focused in reducing inference with a mAP50:95 of 55.9, another in having a good balance between inference and accuracy with a mAP50:95 of 59, and a last one specialized in accuracy with a mAP50:95 of 60.2. Temporal Yolov5 achieves real-time detection in the small and medium architectures. Moreover, it takes advantage of temporal features contained in videos to perform better than Yolov5 in our temporal dataset, making TYolov5 suitable for real-world applications.

Weights of the models:

S ConvLSTM https://drive.google.com/file/d/1AO920AaKX18Ag22E7pmEvoWFZsr2wa6R/view?usp=sharing
M ConvLSTM https://drive.google.com/file/d/120liXvIP71emZeC04_JtLW8ul4DBL8jI/view?usp=sharing
L ConvLSTM https://drive.google.com/file/d/1-sn5QH4SsrooGvyiz9kHwYxsJdpJWpvF/view?usp=sharing
S QRNN https://drive.google.com/file/d/11o10p30ViHBAlv2WqPX-RLKXZRTSf2XH/view?usp=sharing
M QRNN https://drive.google.com/file/d/117ImSTS7bh4phX_85IAe0sVu0kzVYinz/view?usp=sharing
L QRNN https://drive.google.com/file/d/108MaX8mFBVke_7-6qgrXb_9LrOo5U73Y/view?usp=sharing

### If you use this code for your research, please consider citing:

Mario Alberto Duran-Vega, Miguel Gonzalez-Mendoza, Leonardo Chang, Cuauhtemoc Daniel Suarez-Ramirez
https://arxiv.org/abs/2111.08867

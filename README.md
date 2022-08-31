# MDN_SfM
Based on optical flow net and pose odometry net, training a new MobileDetectionNet with self-supervision found out from structure from motion methods, the output is a pixel-level probability map indicating the probability of being a dynamic scene.  This model finally combines with outputs of detectron2.

## Prediction
We train our model use 5 strategies. Three of them are through self-supervised approachs: training with normalized and squared epipolar map(SN mode), training with truncated epipolar map(T mode), training with truncated and gaussian distance weighted epipolar maps(TG mode). Two of them are through semi-supervised approachs: training with Detectron2 network simply combined epipolar maps(DS mode) and combined through cross-entropy similar loss(DC mode). The predicted mobile probability map and their binary masks can be found here.


- The original epipolar map, optical flow and flow error maps of our testset:
https://drive.google.com/drive/folders/13ig6rAVvWrbbbV1oelpMBY7eIOGBwblM?usp=sharing


- Predictions of MDN model:
https://drive.google.com/drive/folders/1DueVT8Fo_6Yn12i3H4nscShNi1QKMgM2?usp=sharing 




## Evaluation 
We generate the binary ground-truth mask from the output of Detectron2, see generate_mobile_gt_d2.py for more details.
The generated ground-truth mask can be found here.
https://drive.google.com/drive/u/0/folders/1HsuBV9__58AVMvc2izzSVEWc4_Ljnp_a 

Based on the generated groud-truth above, we evaluate the mobile probability map of our MobileDetectionNet through 3 metrix: presicion, recall and dice coefficient. You can see the results below.

\documentclass[UTF8]{ctexart}
\begin{document}

\begin{tabular}{ccc}
\hline
姓名& 学号& 性别\\
\hline
Steve Jobs& 001& Male\\
Bill Gates& 002& Female\\
\hline
\end{tabular}

\end{document}

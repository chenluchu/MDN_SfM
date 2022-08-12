# MDN_SfM
Based on optical flow net and pose odometry net, training a new mobile detection net with self-supervision found out from structure from motion methods. 
This model finally combines with outputs of detectron2.

## Prediction
The predicted mobile probability map and their binary masks can be found here.


Prediction by original version of MDN model (training with original epipolar map):


Prediction by second version of MDN model (training with normalized and squared epipolar map):


Prediction by third version of MDN model (training with truncated epipolar map):

Prediction by mix version of MDN model (training with Detectron2 combined epipolar map):
https://drive.google.com/drive/u/0/folders/1Sp6Hp876pEZVzvgKhAF454WZToLGC8ok 


## Evaluation 
We generate the binary ground-truth mask from the output of Detectron2, see generate_mobile_gt_d2.py for more details.
The generated ground-truth mask can be found here.
https://drive.google.com/drive/u/0/folders/1HsuBV9__58AVMvc2izzSVEWc4_Ljnp_a 

Based on the generated groud-truth above, we evaluate the mobile probability map of our MobileDetectionNet through 3 metrix: presicion, recall and dice coefficient. You can see the results below.

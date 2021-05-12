# Underwater-Image-Enhancements
Implementation of Underwater Image Enhancement methods  
Enhancement Methods Used:  
| Enhancement Methods | Link to paper|
| :---: | :---:|
|Integrated Color Model | http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.488.3932&rep=rep1&type=pdf |
|Low Complexity Underwater Image Enhancement Based on Dark Channel Prior| https://ieeexplore.ieee.org/document/6118812 |
|Color Balance and Fusion for Underwater Image Enhancement|https://ieeexplore.ieee.org/document/8058463 |
|Single underwater image restoration by blue-green channels dehazing and red channel correction| https://ieeexplore.ieee.org/document/7471973 |

## Running the code. 
You need to have python3 installed

There has to be images in the file InputImages  

There has to be the following file structure.  
<pre>
.  
|--- InputImages  
|--- OutputImages  
|    |--- Blue_green_channels_dehazing_and_red_channel_correction
|    |--- Color_Balance_and_fusion
|    |--- Integrated_Color_Model
|    |--- Low_Complexity_Enhancement_Based_on_Dark_Channel
|--- Metric_Output
|--- src
|    |--- Blue_green_channels_dehazing_and_red_channel_correction
|    |--- Color_Balance_and_fusion
|    |--- Integrated_Color_Model
|    |--- Low_Complexity_Enhancement_Based_on_Dark_Channel
|--- Makefile
</pre>


### Installing the requirements
```pip install -r requirements.txt```

or 

```pip3 install -r requirements.txt```
### To run the enhancement methods and calculate metrics
```make```
### To run only the enhancement algorithms
```make enhance```
### To only calculate the UCIQE and UIQM
```make metric```

# References 
1. Single-Underwater-Image-Enhancement-and-Color-Restoration https://github.com/wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration
2. Color Balance and Fusion method https://github.com/Zhaofan-Su/DIP_Final

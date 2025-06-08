# Overall Flowchart
![MyPicture](https://raw.githubusercontent.com/Gutsfig/DaHeng_PolCamera/main/img/xiangji.jpg)
# Real-time Display
In a comparison between the Daheng Camera official SDK and our OpenCV-based display, the real-time polarization calculation images can achieve 59 FPS. The processing speed is essentially equivalent to the official DoFP display.
![MyPicture](https://raw.githubusercontent.com/Gutsfig/DaHeng_PolCamera/main/img/tupian.jpg)
# Polarization demosaicing
Demosaic raw polarization image captured with a polarization sensor (e.g., [IMX250MZR/MYR](https://www.sony-semicon.com/en/products/is/industry/polarization.html)).
![MyPicture](https://raw.githubusercontent.com/Gutsfig/DaHeng_PolCamera/main/img/masaike.jpg)
## Introduce
This is a secondary development based on the polarisation camera (MER2-502-79U3M POL), the main functions are real-time display I0, I45, I90, I35, S0, S1, S2, DoLP, AoLP, and real-time saving function.
---
### Development environment
* MSCV2019
* opencv4.9
* cuda11.8
* GalaxySDK
* UTF-8
* Windows
* 4070 Laptop GPU
* i7-12800HX
### Main technology
* Using CUDA to solve mosaic images, real-time solution display can be achieved.
* With multi-threading, pictures can be displayed and saved at the same time.
* All CUDA-based DLL and LIB files have been packaged and uploaded to [Baidu Cloud Drive](https://pan.baidu.com/s/1X-UeQvD5x_L5xcyZwGUeUA?pwd=4grk ) (extraction code: 4grk). Please place the DLL files in the same directory as the generated executable (.exe) file.
---
### Contact
If you have any questions, please contact me at my email 907151833@qq.com.


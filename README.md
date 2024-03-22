## 🤓 Qualitive comparison
|　Input　|　Zero-DCE　|EnGAN|　IAT　|LCDP|　**Ours**　|　GT　|
|---|---|---|---|---|---|---|
|![107_input](./outputs/107_input.png)|![107_zero](./outputs/107_zero.png)|![107_engan](./outputs/107_engan.png)|![107_iat](./outputs/107_iat.png)|![107_lcdp](./outputs/107_lcdp.png)|![107_ours](./outputs/107_ours.png)|![107_gt](./outputs/107_gt.png)|

## 📃 Paper
[[Google drive]](https://drive.google.com/file/d/1Mm4GNnjKdwCFHmbhjJu0HaLLic9oJ2A_/view?usp=sharing)

## 📐 Quantitative comparison

|               | LPIPS   | VIFs    | SSIM    | PSNR     |
| ------------- | ------- | ------- | ------- | -------- |
| Zero-DCE      | 0.254   | 0.512   | 0.756   | 17.383   |
| EnlightenGAN  | 0.231   | 0.500   | 0.768   | 19.187   |
| IAT           | 0.294   | 0.121   | 0.754   | 20.913   |
| LCDP          | *0.160*   | *0.565*   |**0.842**|**23.239**|
| Ours          |**0.157**|**0.567**| *0.832*   | *20.942*   |
* **Bold** means the best and *Italic* means the second.

## 🔥 Method

**Abstract:** 
Fireblight is one of the most devastating diseases for crops such as apples because there is no vaccine. It is necessary to promptly identify such diseases and minimize damage to farmhouses. In addition, the trustworthiness of the model is very important in the discrimination of diseases. In this paper, we propose a methodology to quantify and evaluate which model among several AI models that discriminate specific diseases is the most reliable using Grad-CAM. ResNet50V2, InceptionV3, and Xception were used as three models to discriminate fireblight. All three models showed excellent classification accuracy of 95.87%, 98.38%, and 99.37%. However, the values according to the trustworthiness evaluation function proposed in this paper showed a significant difference at 1.96, 2.42, and 3.51. It was possible to judge that Xception was the most reliable model. The method proposed in this paper can be applied to all models that can use Grad-CAM and enables comparison by quantifying how well the model saw and classified.


## 📂 Dataset

The Dataset is here: [[Google drive]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=146).

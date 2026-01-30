## LDT: Layer-Decomposition Training Makes Networks More Generalizable (ICLR2026)

### [[Paper](https://openreview.net/pdf?id=jLpjcY1iry)] 



Zaizuo Tang<sup>1</sup>, Zongqi Yang<sup>1</sup>, Yubin Yang<sup>1</sup>, 

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China<br>

## Abstract
Domain generalization methods can effectively enhance network performance on test samples with unknown distributions by isolating gradients between unstable and stable parameters. However, existing methods employ relatively coarse-grained partitioning of stable versus unstable parameters, leading to misclassified unstable parameters that degrade network feature processing capabilities. We first provide a theoretical analysis of gradient perturbations caused by unstable parameters. Based on this foundation, we propose Layer-Decomposition Training (LDT), which conducts fine-grained layer-wise partitioning guided by parameter instability levels, substantially improving parameter update stability. Furthermore, to address gradient amplitude disparities within stable layers and unstable layers respectively, we introduce a Dynamic Parameter Update (DPU) strategy that adaptively determines layer-specific update coefficients according to gradient variations, optimizing feature learning efficiency. Extensive experiments across diverse tasks (super-resolution, classification) and architectures (Transformer, Mamba, CNN) demonstrate LDT's superior generalization capability. Our code is available at https://github.com/ZaizuoTang/LDT.



<p align="center">
    <img src="assets/pipeline.png" style="border-radius: 15px">
</p>

⭐If this work is helpful for you, please help star this repo. Thanks!🤗




### Train on SR

1. Download the DRealSR dataset.

    [[DRealSR Dataset](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution)]


2. Download the pre-trained weights.

    [[MambaIR V1 classicSRx4.pth](https://drive.google.com/file/d/1YXggWIsi-auCjmPQDvW9FjB1f9fZK0hN/view?usp=sharing)]


3. Start training:

    python Train_pip.py


### Test on SR

    python basicsr/test.py -opt test_MambaIR_SR_x4.yml






## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [MambaIR](https://github.com/csguoh/MambaIR). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at tangzz@smail.nju.edu.cn

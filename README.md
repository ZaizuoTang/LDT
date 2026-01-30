# LDT: Layer-Decomposition Training Makes Networks More Generalizable (ICLR2026)

This repository is an official PyTorch implementation of the paper [LDT: Layer-Decomposition Training Makes Networks More Generalizable](https://openreview.net/pdf?id=jLpjcY1iry).

Zaizuo Tang<sup>1</sup>, Zongqi Yang<sup>1</sup>, Yubin Yang<sup>1</sup>, 

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China<br>

## Abstract
Domain generalization methods can effectively enhance network performance on test samples with unknown distributions by isolating gradients between unstable and stable parameters. However, existing methods employ relatively coarse-grained partitioning of stable versus unstable parameters, leading to misclassified unstable parameters that degrade network feature processing capabilities. We first provide a theoretical analysis of gradient perturbations caused by unstable parameters. Based on this foundation, we propose Layer-Decomposition Training (LDT), which conducts fine-grained layer-wise partitioning guided by parameter instability levels, substantially improving parameter update stability. Furthermore, to address gradient amplitude disparities within stable layers and unstable layers respectively, we introduce a Dynamic Parameter Update (DPU) strategy that adaptively determines layer-specific update coefficients according to gradient variations, optimizing feature learning efficiency. Extensive experiments across diverse tasks (super-resolution, classification) and architectures (Transformer, Mamba, CNN) demonstrate LDT's superior generalization capability. Our code is available at https://github.com/ZaizuoTang/LDT.




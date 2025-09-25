# [ICCV 2025] MedVSR: Medical Video Super-Resolution with Cross State-Space Propagation

**Xinyu Liu<sup>1</sup>, Guolei Sun<sup>2</sup>, Cheng Wang<sup>1</sup>, Yixuan Yuan<sup>1,\*</sup>, Ender Konukoglu<sup>2,\*</sup>**  
<sup>1</sup>The Chinese University of Hong Kong  
<sup>2</sup>Computer Vision Laboratory, ETH Zurich  

---

## Overview
**MedVSR** is a tailored model for medical VSR. 
It first employs Cross State-Space Propagation (CSSP) to address the imprecise alignment by projecting distant frames as control matrices within state-space models, enabling the selective propagation of consistent and informative features to neighboring frames for effective alignment.
It also features an Inner State-Space Reconstruction (ISSR) module that enhances tissue structures and reduces artifacts with joint long-range spatial feature learning and large-kernel short-range information aggregation.

---

## Installation

Clone this repository:
```bash
git clone https://github.com/CUHK-AIM-Group/MedVSR
cd MedVSR

conda create -n MedVSR python==3.9
conda activate MedVSR

pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pip install -e causal_conv1d>=1.1.0
pip install -e mamba-1p1p1
```

---

## Dataset preparation

For the preprocessed HyperKvasir, LDPolyp, and EndoVis18, please download from [huggingface link](https://huggingface.co/datasets/jeffrey423/MedVSR_dataset). Modify L14-16 and L39-40 to the extracted HyperKvasir training and validation folders.

## Test the model

Download our pretrained model at [here](https://huggingface.co/jeffrey423/MedVSR).

```python
python test_model.py -opt ./options/medvsr_train.yml --weight <PATH_TO_PRETRAINED_MEDVSR>
```

## Training 
```bash
bash dist_train.sh 2 options/medvsr_train.yml 25623
```

## Citation
```bibtex
@inproceedings{liu2025medvsr,
  title     = {MedVSR: Medical Video Super-Resolution with Cross State-Space Propagation},
  author    = {Liu, Xinyu and Sun, Guolei and Wang, Cheng and Yuan, Yixuan and Konukoglu, Ender},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```


## Acknowledgement

We sincerely thank the authors and contributors of the following projects for their awesome codebases, which have greatly benefited our work:

- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [IART](https://github.com/kai422/IART)
- [RVRT](https://github.com/JingyunLiang/RVRT)
- [Mamba](https://github.com/state-spaces/mamba)
- [MambaVision](https://github.com/NVlabs/MambaVision)
- [Vim](https://github.com/hustvl/Vim)

## Contact

Please contact [xinyuliu@link.cuhk.edu.hk](mailto:xinyuliu@link.cuhk.edu.hk) or open an issue.
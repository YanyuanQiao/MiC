# MiC
Code of the ICCV 2023 paper :
<br>**March in Chat: Interactive Prompting for Remote Embodied Referring Expression**<br>

[[Paper](https://arxiv.org/pdf/2308.10141.pdf)] [[GitHub](https://github.com/YanyuanQiao/MiC)]

![teaser](overview.png)

</p>

## Abstract
Many Vision-and-Language Navigation (VLN) tasks have been proposed in recent years, from room-based to object-based and indoor to outdoor. The REVERIE (Remote Embodied Referring Expression) is interesting since it only provides high-level instructions to the agent, which are closer to human commands in practice. Nevertheless, this poses more challenges than other VLN tasks since it requires agents to infer a navigation plan only based on a short instruction. Large Language Models (LLMs) show great potential in robot action planning by providing proper prompts. Still, this strategy has not been explored under the REVERIE settings. There are several new challenges. For example, the LLM should be environment-aware so that the navigation plan can be adjusted based on the current visual observation. Moreover, the LLM planned actions should be adaptable to the much larger and more complex REVERIE environment. This paper proposes a March-in-Chat (MiC) model that can talk to the LLM on the fly and plan dynamically basexzd on a newly proposed Room-and-Object Aware Scene Perceiver (ROASP). Our MiC model outperforms the previous state-of-the-art by large margins by SPL and RGSPL metrics on the REVERIE benchmark.

## TODOs

- [x] Demonstration Set
- [x] ROASP Code
- [ ] LLM Code 

## Prerequisites
### Installation and Data Preparation
Please follow the installation instructions in [DUET](https://github.com/cshizhe/VLN-DUET) to set up the environment.
#### Trained Weights
Download the trained weights from [here](https://drive.google.com/drive/folders/1LfV3DBYcS_pS2rxAIYOX8sRGLdGFQYN7?usp=drive_link).
### Room-and-Object Aware Scene Perceiver (ROASP)
```sh
cd ROASP
```
To predict the room type
```sh
python process-clip-room-ViTB32.py
```
To predict the object type
```sh
python process-clip-obj-ViTB32.py
```
## Citation
Please cite our paper:
```
@InProceedings{Qiao_2023_MiC,
    author    = {Qiao, Yanyuan and Qi, Yuankai and Yu, Zheng and Liu, Jing and Wu, Qi},
    title     = {March in Chat: Interactive Prompting for Remote Embodied Referring Expression},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {15758-15767}
}
```

## Acknowledgement

We thank the developers of [DUET](https://github.com/cshizhe/VLN-DUET),  [language-planner](https://github.com/huangwl18/language-planner) for their public code release.

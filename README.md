# CI-Diff
This repository is the official code for the paper "Rare Concept Generation via Counterfactual Inference in Diffusion Models" by Zhengyuan Jiang (2024110489@mail.hfut.edu.cn), Haipeng Liu(equal contribution: hpliu_hfut@hotmail.com), Meng Wang, Yang Wang(corresponding author: yangwang@hfut.edu.cn). ACM Multimedia 2026, Rio de Janeiro, Brazil.

# Inference
1.  Dataset Preparation: [Rarebench](https://github.com/krafton-ai/Rare-to-Frequent).
2.  Extend categories: Style categories See `datasets/single_6style.txt` folder and scen categories See `datasets/single_8scen.txt` folder.
3.  Pre-trained models: [stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large); [ip-adapter.bin](https://huggingface.co/h94/IP-Adapter); [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384).
4.  Run the following command:
   ```bash
Python test.py
   ```
# Example Results
- Visual comparison between our method and the competitors.
  
![Example Results](image/image0.png)

- Quantitative results(C denotes CLIP-T score; H denotes HPSv2 score; L denotes LLM score and U denotes User Study)

![Example Results](image/image1.png)
![Example Results](image/image2.png)

- Ablation Studies

![Example Results](image/image3.png)
![Example Results](image/image4.png)

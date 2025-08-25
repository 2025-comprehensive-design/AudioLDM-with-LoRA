<div align="center">

# AudioLDM with LoRA

### **Enhancing Diffusion-Based Music Generation Performance with LoRA**.

Kim, Seonpyo, Geonhui Kim, Shoki Yagishita, Daewoon Han, Jeonghyeon Im, and Yunsick Sung. 2025. "Enhancing Diffusion-Based Music Generation Performance with LoRA" Applied Sciences 15, no. 15: 8646. https://doi.org/10.3390/app15158646

[![MDPI – Applied Sciences](https://img.shields.io/badge/MDPI-Applied%20Sciences-1D4B8F.svg?style=flat-square)](https://www.mdpi.com/2076-3417/15/15/8646)

</div>

## Overview

This repository contains code for fine-tuning the Text-to-Audio model, AudioLDM, using the LoRA ([LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) ) technique, as described in our paper.

---

## Abstract

Recent advancements in generative artificial intelligence have significantly progressed the field of text-to-music generation, enabling users to create music from natural language descriptions. Despite the success of various models, such as MusicLM, MusicGen, and AudioLDM, the current approaches struggle to capture fine-grained genre-specific characteristics, precisely control musical attributes, and handle underrepresented cultural data. This paper introduces a novel, lightweight fine-tuning method for the AudioLDM framework using low-rank adaptation (LoRA). By updating only selected attention and projection layers, the proposed method enables efficient adaptation to musical genres with limited data and computational cost. The proposed method enhances controllability over key musical parameters such as rhythm, emotion, and timbre. At the same time, it maintains the overall quality of music generation. This paper represents the first application of LoRA in AudioLDM, offering a scalable solution for fine-grained, genre-aware music generation and customization. The experimental results demonstrate that the proposed method improves the semantic alignment and statistical similarity compared with the baseline. The contrastive language–audio pretraining score increased by 0.0498, indicating enhanced text-music consistency. The kernel audio distance score decreased by 0.8349, reflecting improved similarity to real music distributions. The mean opinion score ranged from 3.5 to 3.8, confirming the perceptual quality of the generated music.

Keywords: text-to-music generation; Parameter-Efficient Fine-Tuning (PEFT); low-rank adaptation (LoRA)


## Project Structure

```
.
├── app.py              # Gradio application for audio generation inference
├── requirements.txt    # Python dependencies
├── config/             # Configuration files (config.yaml, dataConfig.yaml)
├── data/               # Directory for storing audio datasets
├── notebook/           # Jupyter notebooks for examples and experiments
└── script/             # Core scripts for the pipeline
    ├── data/           # Scripts for data preprocessing
    ├── train/          # Scripts for model training and fine-tuning
    ├── inference/      # Scripts for running inference
    ├── utilities/      # Helper scripts
    └── push_to_hub.py  # Script to upload models to Hugging Face Hub
```

---

## Installation

1.  **Create and activate a conda environment:**

    ```bash
    conda create -n audioldm python=3.8
    conda activate audioldm
    ```

2.  **Install dependencies:**
    ```bash
    # Install AudioLDM
    pip3 install git+https://github.com/haoheliu/AudioLDM.git
    ```

    Clone the repository and install the required packages from `requirements.txt`.

    ```bash
    git clone <repository-url>
    cd AudioLDM-with-LoRA
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Configuration

-   Modify `config/config.yaml` to set training parameters like learning rate, epochs, etc.
-   Modify `config/dataConfig.yaml` to specify paths to your dataset and other data-related settings.

### 2. Data Preparation

-   Use the scripts in `script/data/` to preprocess your audio files and create a dataset suitable for training. Example usage can be found in the notebooks.

### 3. Training

-   Run the training scripts located in `script/train/` to fine-tune the AudioLDM model with LoRA.
-   Training progress and logs will be saved to the specified output directories.

### 4. Inference

-   Once the model is trained, you can generate audio from text prompts.
-   The `app.py` file provides a Gradio interface for easy inference. It loads the base model and your fine-tuned LoRA weights.

    ```shell
    python app.py
    ```

-   You can also use the scripts in `script/inference/` for command-line inference.

### 5. Share to Hub

-   Use the `script/push_to_hub.py` script to upload your trained LoRA weights to the Hugging Face Hub.

---

## Model

-   **Base Model:** This project uses [`cvssp/audioldm-s-full-v2`](https://huggingface.co/cvssp/audioldm-s-full-v2) as the base model for fine-tuning.

## Citation

If you find this work useful, please cite our paper:

<div>

Kim, S., Kim, G., Yagishita, S., Han, D., Im, J., & Sung, Y. (2025). Enhancing Diffusion-Based Music Generation Performance with LoRA. Applied Sciences, 15(15), 8646. https://doi.org/10.3390/app15158646

</div>

---
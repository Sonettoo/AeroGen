# AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation

<a href='https://arxiv.org/abs/2411.15497'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href=#bibtex><img src='https://img.shields.io/badge/Paper-BibTex-Green'></a> 
<a href='https://huggingface.co/Sonetto702/AeroGen'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DiffBlender_Model-blue'></a>


- **AeroGen** AeroGen is the first model to simultaneously support horizontal and rotated bounding box condition generation, thus enabling the generation of high-quality synthetic images that meet specific layout and object category requirements.

<div align=center>
<img src="imgs/display.png" height="100%" width="100%"/>
</div>

## 🗓️ TODOs

- [x] Release pretrained models.
- [x] Release inference code.
- [x] Release training code
- [ ] Release Gradio UI.

## 🚀 Getting Started

### Conda environment setup
prepare the environment

```bash
conda env create -f environment.yaml
conda activate aerogen
```
You can download pre-trained models from this [huggingface url](https://huggingface.co/Sonetto702/AeroGen) and put it to `./ckpt/` folder.

### ⚡️Quick Generation

You can the following code to generate images more quickly by:
```bash
python src/test/inference/inference.py
```
You can find the relevant layout files for the presentation in `./demo/` where you can find the relevant layout files for the display.
The following is the example of the generated image.
<div align=center>
<img src="imgs/display1.png" height="80%" width="80%"/>
</div>

## Training Datasets Preperation
We use the DIOR-R dataset as an example to show how to set training dataset.
Download DIOR-R dataset from [url](https://gcheng-nwpu.github.io/) and save in `./datasets/`. 
```
├── datasets
│   ├── DIOR-VOC
│   │   ├── Annotations
│   │   │   ├── Oriented_Bounding_Boxes
│   │   │       ├── ... (annotation files, e.g., .xml)
│   │   ├── VOC2007
│   │   │   ├── JPEGImages
│   │   │   │   ├── ... (image files, e.g., .jpg, .png)
│   │   │   ├── ImageSets
│   │   │   │   ├── Main
│   │   │   │       ├── train.txt
│   ├── category_embeddings.npy
```


## 🎶 Model Training

The following demonstrates the model training process under the DIOR-R dataset, firstly preparing the pytorch environment and the training dataset in [DATASETS](datasets/README.md), then downloading the SD weights fine-tuned on remote sensing images to the ckpt folder at this [url](https://huggingface.co/Sonetto702/AeroGen) & put it to `./ckpt/`, and finally executing the following commands in sequence:

```bash
conda activate aerogen
python src/train/prepare_weight_r.py
bash configs/stable-diffusion/hrsc/v1-finetune-DIOR-R.yaml
```
The more information and options an find in `./main.py` and `./configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml`

## 📡 Contact
If you have any questions about the paper or the code, feel free to email me at [aryswph@gmail.com](mailto:aryswph@gmail.com). This ensures I can promptly notice and respond!

## 💕 Acknowledgments:
This repo is built upon [Stable Diffusion](https://github.com/CompVis/stable-diffusion/tree/main), [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main), [CLIP](https://github.com/openai/CLIP), [GLIGEN](https://github.com/gligen/GLIGEN/tree/master). Sincere thanks to their excellent work!


## Citation
```
@article{tang2024aerogen,
  title={AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation},
  author={Tang, Datao and Cao, Xiangyong and Wu, Xuan and Li, Jialin and Yao, Jing and Bai, Xueru and Meng, Deyu},
  journal={arXiv preprint arXiv:2411.15497},
  year={2024}
}
```

# ANN & Deep Learning — Assignment 02  
## Fine-tuning ResNet-50 and U-Net on Small Image Datasets

Transfer learning on two small datasets:

| Part | Model | Task | Dataset (subset) | Main metric |
|------|--------|------|------------------|-------------|
| A | ResNet-50 (ImageNet pretrained) | Classification | Oxford Flowers-102 (5×40 = 200 images) | Accuracy / F1 |
| B | U-Net + ResNet-50 encoder | Binary segmentation | Oxford-IIIT Pet (150 image–mask pairs) | Dice / IoU |

Both models use a **two-phase** fine-tuning strategy: freeze the pretrained backbone/encoder first, then unfreeze the last stages with a smaller learning rate.

---

## Repository contents

```text
├── Assignment02.ipynb      # main notebook (run top-to-bottom, with outputs)
├── prepare_datasets.py     # builds the small subsets from official archives
├── results/                # figures (confusion matrix, curves, predictions)
├── .gitignore
└── README.md
```

**Not included** (too large / not required): raw datasets, trained `.pth` weights, local `.venv`.

---

## Datasets

### Official downloads
- Flowers: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/  
  Files: `102flowers.tgz`, `imagelabels.mat`
- Pets: https://www.robots.ox.ac.uk/~vgg/data/pets/  
  Files: `images.tar.gz`, `annotations.tar.gz`

### Expected folder layout (after preparation)
```text
ANN_DL_Assignment/
├── flowers_subset/
│   ├── flower_class_000/ ...
│   └── flower_class_004/
└── pets_subset/
    ├── images/   *.jpg
    └── masks/    *.png   # same stem as image (trimap → binary in code)
```

You can also upload this folder to Google Drive as `MyDrive/ANN_DL_Assignment/` for Colab.

To rebuild subsets locally (optional):

```bash
python prepare_datasets.py
```

---

## Environment

- Python 3.10+ recommended  
- GPU recommended (this work was run on an **NVIDIA RTX 4070**)  
- Main packages: `torch`, `torchvision`, `segmentation-models-pytorch`, `scikit-learn`, `matplotlib`, `pillow`, `scipy`

Example (CUDA PyTorch — adjust CUDA tag if needed):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install segmentation-models-pytorch scikit-learn matplotlib pandas pillow scipy tqdm ipykernel
```

Open `Assignment02.ipynb` and select the same environment as the kernel.

---

## How to run

1. Prepare `ANN_DL_Assignment/` (or mount Drive on Colab).  
2. Set `BASE_DIR` in the notebook path cell to your folder.  
3. Run all cells **top to bottom**.  
4. Checkpoints are saved under `checkpoints/`; figures under `results/`.

**Splits:** flowers stratified **70% / 15% / 15%**; pets shuffled **70% / 15% / 15%**.  
Validation is used for model selection; the **test set is used once** at the end.

---

## Results (this submission)

### ResNet-50 (flowers)
| Metric | Value |
|--------|------:|
| Training accuracy | 1.000 |
| Validation accuracy | 1.000 |
| **Test accuracy** | **0.933** |
| Precision / Recall / F1 (macro) | 0.950 / 0.933 / **0.935** |

Phase 1 (frozen backbone) best val ≈ 0.90 → Phase 2 (`layer4` fine-tune) improved validation further.  
Test is slightly below val because the val set is small (30 images); **test accuracy is the honest score**.

### U-Net (pets)
| Metric | Value |
|--------|------:|
| Training loss | 0.114 |
| Validation Dice / IoU | 0.927 / 0.868 |
| **Test Dice / IoU** | **0.936 / 0.881** |
| Test pixel accuracy | 0.954 |

Phase 1 (frozen encoder) best val Dice ≈ 0.91 → Phase 2 (unfreeze encoder `layer3`+`layer4`) best val Dice ≈ 0.93.  
Train / val / test stay close — no severe over- or underfitting.

Figures: see the `results/` folder and notebook outputs.

---

## Method notes (short)

- **Why transfer learning?** Small labelled sets; ImageNet features transfer well.  
- **Why freeze then fine-tune?** New head/decoder trains first; later layers adapt gently with a lower LR.  
- **Why no augmentation on val/test?** Evaluation must be stable and comparable.  
- **Why BCE + Dice for segmentation?** BCE per pixel + Dice for overlap / class imbalance.  
- **Why NEAREST resize for masks?** Keeps mask labels as 0/1 (no blurred borders).

---

## Author

- Name: __________________  
- Roll No.: __________________  

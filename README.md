# Image Segmentation Review: U-Net & DeepLabV3

**Notebook:** `ce6190-segmentation-review-unet-deeplabv3`

Compact, reproducible segmentation evaluation comparing **DeepLabV3** and **U-Net** on small, easy-to-run datasets (Cityscapes subset, Kvasir-SEG, Supervisely Person). The notebook trains/evaluates models, runs a small ablation suite, and produces the plots/tables needed for a short report.


## Repository contents

- `ce6190-segmentation-review-unet-deeplabv3.ipynb`: Main notebook (training, evaluation, ablations, cross-eval, plotting).  
- `Image_Segmentation_Review_DeepLabV3_and_U_Net.pdf`: The compiled paper of the segmentation review.



## Quick summary

The notebook can:

- Train and evaluate **U-Net** and **DeepLabV3** with configurable hyperparameters (backbone, input size, optimizer, loss).
- Work on three datasets: **Cityscapes (subset)**, **Kvasir-SEG**, **Supervisely Person**.
- Save/load checkpoints and produce evaluation outputs (mIoU, per-class IoU, Dice, pixel accuracy).
- Run prioritized **ablations** (backbone depth, output stride, encoder init, input resolution).
- Run cross-dataset generalization (Kvasir-SEG → Supervisely Person).
- Generate plots/tables for reports.



## Installation

- **Install core dependencies**
   All main Python packages used in the notebook are listed in:

   ```
   requirements.txt
   ```

   Install them with:

   ```bash
   pip install -r requirements.txt
   ```

- **Install PyTorch (IMPORTANT, choose your CUDA version)**
   PyTorch must be installed using the official wheels to match your system’s CUDA version.

   Go to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

   Example (CUDA 11.8, modify for your system):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   If you do not have a GPU, install CPU-only:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

- **Start JupyterLab**

   ```bash
   jupyter lab
   ```

   Then open:

   ```
   ce6190-segmentation-review-unet-deeplabv3.ipynb
   ```


- **Download required assets**

   * **Pretrained models & results archive** (used by the notebook):
     `https://www.kaggle.com/datasets/ramiaridi/ce6190-segmentation-review-unet-deeplabv3-dataset`
   * **Datasets** (mirrors used in the notebook):

     * Cityscapes (subset): `https://www.kaggle.com/datasets/xiaose/cityscapes`
     * Kvasir-SEG: `https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation`
     * Supervisely Person: `https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset`

   Modify the path in the `Config` section of the notebook accordingly.
   ```python
    models_pth = "/kaggle/input/ce6190-segmentation-review-unet-deeplabv3-dataset/trained-models-weights/trained-models-weights" # To load pretrained Models


    if DATASET == "Cityscapes":
        base_dir = "/kaggle/input/cityscapes/Cityspaces"
    elif DATASET == "Supervisely":
        base_dir = "/kaggle/input/supervisely-filtered-segmentation-person-dataset"
    elif DATASET == "Kvasir":
        base_dir = "/kaggle/input/kvasir-dataset-for-classification-and-segmentation/kvasir-seg/Kvasir-SEG"
    else:
        print("Choose DATASET = <Cityscapes> OR <Supervisely> OR <Kvasir>")
   
   ```


## Notebook configuration (example)

In the notebook config cell you can set:

```python
TRAIN = False            # False = evaluation only; True = train models
DATASET = "Kvasir"       # choices: "Cityscapes", "Kvasir", "Supervisely"
MODEL = "DeeplabV3"      # choices: "DeeplabV3", "Unet"
```

Make sure the `model` variable is updated appropriately in evaluation/visualization cells, e.g.:

```python
model = deeplabv3_cityscapes
```

See the **Experiment table** below to map names to files.

---


## Experiment table & canonical model names

| Model Name                 | Model      | TrainedOn  | EvalOn     | Backbone  | Pretrained | OutputStride | InputSize | Notes                 |
| -------------------------- | ---------- | ---------- | ---------- | --------- | ---------- | ------------ | --------- | --------------------- |
| deeplabv3_cityscapes       | DeepLabV3 | Cityscapes | Cityscapes | resnet50  | imagenet   | 8            | 256       | Baseline              |
| deeplabv3_101_cityscapes   | DeepLabV3 | Cityscapes | Cityscapes | resnet101 | imagenet   | 8            | 256       | Ablation (backbone)   |
| deeplabv3_os16_cityscapes  | DeepLabV3 | Cityscapes | Cityscapes | resnet50  | imagenet   | 16           | 256       | Hyperparameter (OS)         |
| deeplabv3_kvasir           | DeepLabV3 | Kvasir     | Kvasir     | resnet50  | imagenet   | 8           | 256       | Baseline              |
| unet_cityscapes            | U-Net      | Cityscapes | Cityscapes | resnet50  | imagenet   | N/A          | 256       | Baseline              |
| unet_noimagenet_cityscapes | U-Net      | Cityscapes | Cityscapes | resnet50  | None       | N/A          | 256       | Ablation (init)       |
| unet_kvasir                | U-Net      | Kvasir     | Kvasir     | resnet50  | imagenet   | N/A          | 256       | Baseline              |
| unet_512_kvasir            | U-Net      | Kvasir     | Kvasir     | resnet50  | imagenet   | N/A          | 512       | Hyperparameter (input size) |

---

## Ablations, hyperparameter & cross-eval

**DeepLabV3**

* Backbone depth: `resnet50` → `resnet101`
* Output stride: `OS=8` → `OS=16`

**U-Net**

* Encoder init: `imagenet` → `None` (random)
* Input size: `256` → `512`

**Cross-evaluation**

* `Train: Kvasir-SEG` → `Eval: Supervisely Person` (measure domain generalization)


## Metrics

Computed per-model/run:

* **mIoU (mean Intersection over Union)** — primary metric
* **Per-class IoU**
* **Dice coefficient (Sørensen–Dice / F1)** — per-image and per-class
* **Pixel accuracy**
* Per-image IoU lists saved for statistical testing/plots


## Reproducibility tips

* Preserve dataset folder structure. If you move files, update the notebook path variables.
* Place pretrained weights in `models/` and match filenames to the experiment table.
* Set random seeds in the notebook to aid reproducibility.
* Use a GPU for training when available; CPU runs are possible but slow.


## Dependencies (high-level)

* Python 3.8+ recommended
* `numpy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`, `ipywidgets`
* `torch`, `torchvision`
* `segmentation-models-pytorch` (SMP)
* Jupyter / JupyterLab to run the notebook


## Contact

Questions / issues: open an issue or contact **Rami Aridi**

GitHub: `RamiAridi03`

---

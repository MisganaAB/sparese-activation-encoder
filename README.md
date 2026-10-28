# Sparse Autoencoders on GPT-2 Small — Mechanistic Interpretability Research

This research repository investigates the properties of Sparse Autoencoders (SAEs) applied to internal residual-stream representations of pretrained **GPT-2 Small**.

## Research Question
> **How does changing the sparsity penalty of a sparse autoencoder affect its ability to reconstruct activations from GPT-2 Small, the number and frequency of active latents, and the contexts that activate learned features?**

---

## 1. Project Structure

```
├── .gitignore                          # Version control exclusions
├── README.md                           # Project documentation & execution guide
├── requirements.txt                    # Pinned, tested Python dependencies
├── checkpoints/                        # Model & activation cache storage
│   ├── cached_activations/             # WikiText-2 activation tensors & metadata
│   │   ├── train_acts.pt               # [8312, 768] training activations
│   │   ├── val_acts.pt                 # [2049, 768] validation activations
│   │   ├── test_acts.pt                # [2112, 768] held-out test activations
│   │   └── test_meta.json              # 2,112 token context metadata records
│   └── sample_cache/                   # Phase 1 verification cache & metadata
├── data/
│   └── README.md                       # Corpus splits and anti-leakage guidelines
├── notebooks/
│   ├── 00_environment_setup.ipynb      # Phase 1: Environment & model verification
│   └── 01_sae_experiment.ipynb         # Phase 2–5: Caching, SAE training & comparative eval
├── outputs/
│   └── sae/
│       ├── sae_run_a_lambda_0.001.pt   # Checkpoint for Run A (Low Sparsity)
│       ├── sae_run_b_lambda_0.01.pt    # Checkpoint for Run B (High Sparsity)
│       ├── sparsity_comparison.png     # Validation MSE & activation frequency plot
│       ├── metrics_summary.json        # Quantitative evaluation metrics
│       └── inspected_features.json     # Top-activating contexts for inspected latents
└── report/
    └── research_report.md              # Complete 900-word empirical research report
```

---

## 2. Verified Empirical Results

Both SAE runs ($d_{in} = 768, d_{sae} = 1024$) were trained on the identical 8,312 WikiText-2 Layer 6 residual stream activations (`blocks.6.hook_resid_post`) and evaluated on 2,112 held-out test activations:

| Metric | Run A ($\lambda = 0.001$) | Run B ($\lambda = 0.01$) | Impact of $10\times$ Sparsity Penalty |
| :--- | :---: | :---: | :---: |
| **Reconstruction MSE** | 0.8865 – 0.9383 | 2.9090 – 2.9148 | Controlled increase in residual reconstruction error |
| **Variance Accounted For (VAF)** | **99.50% – 99.53%** | **98.48%** | **Remains exceptionally high (> 98.4%)** |
| **Fraction of Variance Explained (FVE)** | **99.49% – 99.52%** | **98.42% – 98.43%** | High reconstruction fidelity across all dimensions |
| **Mean Active Latents ($L_0$)** | **481.28 – 484.54** | **41.12 – 44.37** | **$11.7\times$ reduction in active features per token** |
| **Median Active Latents ($L_0$)** | 491.0 – 495.0 | 35.0 | Strong leftward shift in latent activity |
| **Dead Latent Rate (0 fires on test)** | **0.0%** (0 / 1024) | **0.0%** (0 / 1024) | Zero feature collapse |

---

## 3. Google Colab Execution Guide

### Step 1: Open the Notebooks in Google Colab
1. Upload this repository or notebooks to GitHub or Google Drive.
2. Open [Google Colab](https://colab.research.google.com).
3. Select **File > Open Notebook** and load:
   - First: `notebooks/00_environment_setup.ipynb`
   - Second: `notebooks/01_sae_experiment.ipynb`

### Step 2: Choose Hardware Accelerator
- In Colab, navigate to **Runtime > Change runtime type**.
- Select **T4 GPU** (or standard CPU). The code dynamically handles both:
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```

### Step 3: Checkpointing & Saving Results Before Runtime Reset
Google Colab runtimes reset after inactivity. Both notebooks include automated backup cells:

#### Option A: Direct Google Drive Mount
```python
from google.colab import drive
drive.mount('/content/drive')
# Backs up the entire outputs/ and checkpoints/ folders
import shutil
shutil.copytree("outputs", "/content/drive/MyDrive/sae_outputs", dirs_exist_ok=True)
shutil.copytree("checkpoints", "/content/drive/MyDrive/sae_checkpoints", dirs_exist_ok=True)
```

#### Option B: Browser Archive Download
```python
from google.colab import files
import shutil
shutil.make_archive("sae_experiment_artifacts", "zip", "outputs")
files.download("sae_experiment_artifacts.zip")
```

---

## 4. Local Installation & Verification

To execute locally:
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter lab
```

---

## 5. Dependency Compatibility Notes
- **TransformerLens 4.x Migration:** TransformerLens 4.0 removed `HookedTransformer` in favor of `TransformerBridge`. To ensure compatibility with standard mechanistic interpretability code and SAE Lens patterns, `requirements.txt` pins `transformer-lens>=2.0.0,<3.0.0` with `transformers>=4.40.0,<5.0.0`.
- **Parameter Freezing:** GPT-2 Small parameters are strictly set to `requires_grad = False` and evaluated in `model.eval()`. No language model fine-tuning occurs.

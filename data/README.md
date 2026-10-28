# Sparse Autoencoders on GPT-2 Small — Data Documentation

## 1. Corpus Selection: WikiText-2
This research project uses the standard **WikiText-2** benchmark corpus (`wikitext-2-raw-v1` via Hugging Face `datasets`) to extract internal activations from GPT-2 Small for Sparse Autoencoder (SAE) training and evaluation.

### Why WikiText-2?
- **Natural Language Diversity:** Derived from verified Wikipedia featured articles, providing a balanced distribution of syntactic structures, factual knowledge, and semantic domains.
- **Reproducibility:** A standard, open-access NLP benchmark with stable, canonical document splits.
- **Resource Efficiency:** Well-suited for Google Colab runtimes and local CPU/GPU execution without hitting memory bottlenecks.

---

## 2. Document Partitioning & Anti-Leakage Hygiene

To ensure rigorous evaluation without data contamination, the corpus is partitioned at the **document level prior to tokenization and activation extraction**:
1. **Training Split (`train`):** Used strictly to extract activations for SAE dictionary learning ($W_{\text{enc}}, W_{\text{dec}}, b_{\text{enc}}, b_{\text{dec}}$).
2. **Validation Split (`validation`):** Used for monitoring reconstruction loss and $L_0$ sparsity during training and hyperparameter checks.
3. **Held-Out Test Split (`test`):** Strictly isolated held-out set reserved for final comparative evaluation of both SAE runs (measuring reconstruction MSE, Fraction of Variance Explained / VAF, active latent counts, dead feature rates, and top-activating contexts).

> **Crucial Rule:** Activations from the `test` split are NEVER seen during SAE training or hyperparameter tuning.

---

## 3. Activation Cache Structure & Metadata Format

Activations are extracted from a documented GPT-2 Small residual-stream site (`blocks.6.hook_resid_post`, dimension $d_{\text{model}} = 768$).

### Preprocessing & Filtering:
- **No Padding Pollution:** Padded positions are strictly excluded; only genuine token activations are stored in the training/eval buffers.
- **Metadata Traceability:** Each activation tensor is stored alongside metadata that records:
  - `doc_id` / document index
  - `token_position` within the document
  - `token_string` (exact text representation)
  - `token_id` (vocabulary integer ID)

### Serialization Format:
- **Tensor Storage:** `checkpoints/sae_activations_train.pt` (CPU-detached float32 tensor of shape `[N_tokens, 768]`).
- **Metadata Storage:** `checkpoints/sae_activations_train_meta.json` (JSON list recording token text, positions, and split details).
- Storing detached CPU tensors ensures seamless loading across GPU and CPU environments and eliminates repeating forward passes through GPT-2 Small.

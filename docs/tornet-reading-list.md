# TorNet Hybrid Reading List

Papers and architecture notes for a CNN–transformer hybrid built on the TorNet CNN baseline: the dataset, the convolutional pieces it reuses, and the transformer designs it borrows from.

---

## 1. Architecture spectrum

```
◄ PURE CONVOLUTION ─────────────────────────────────────────── PURE ATTENTION ►

      CNN               ViT-hybrid              CvT                  ViT
       ●─────────────────────◉─────────────────────●────────────────────●
  TorNet baseline      YOUR DESIGN        conv inside           patches →
                   CNN features →         transformer          transformer
                    transformer
```

| Architecture | How it works | Strength | Weakness |
|---|---|---|---|
| **CNN** | Stacked convolutions and pooling | Data-efficient, strong locality bias | Limited global context |
| **ViT-hybrid** ⭐ | CNN backbone, then a transformer over its feature map | Reuses a pretrained CNN, adds global attention | Two separate parts; needs positional embeddings |
| **CvT** | Transformer with conv token embeddings and conv Q/K/V | Hierarchical; no positional embeddings needed | More complex to build from scratch |
| **ViT** | Image split into patches, then a pure transformer | Global attention from the first layer | Data-hungry, weak inductive bias |

---

## 2. TorNet CNN baseline: actual architecture

Taken from `tornet/models/keras/cnn_baseline.py` → `build_model()`. Where the code and the paper disagree, the **code is what the pretrained weights match**.

```
INPUTS (each 120 azimuth × 240 range × 2 sweeps)
  6 radar variables ─► Normalize_<var> ─┐
                                        ├─► Concatenate ─► FillNaNs(-3) ─► + range_folded_mask
  coordinates (r, 1/r) ─────────────────┼────────────────────────────────────────┐
                                        ▼                                        ▼
┌─────────────────────────── KEEP: feature extractor ───────────────────────────┐
│ Block 1: 2× CoordConv 3×3, 64 filters, ReLU → MaxPool → Dropout   → 60×120    │
│ Block 2: 2× CoordConv 3×3, 128 filters     → MaxPool → Dropout    → 30×60     │
│ Block 3: 3× CoordConv 3×3, 256 filters     → MaxPool → Dropout    → 15×30     │
│ Block 4: 3× CoordConv 3×3, 512 filters     → MaxPool → Dropout    → 7×15      │
└───────────────────────────────────────────────────────────────────────────────┘
                                        ▼  feature map: 7 × 15 × 512
┌──────────────────────────── CUT: head='maxpool' ──────────────────────────────┐
│ 1×1 Conv, 512, ReLU                                                           │
│ 1×1 Conv, 256, ReLU                                                           │
│ 1×1 Conv, 1  ("heatmap", linear) → per-location tornado logits                │
│ GlobalMaxPooling2D               → single logit                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### What you keep vs. cut

| Part | Keep or cut | Why |
|---|---|---|
| Normalization, NaN fill, range-folded mask | **Keep** | Inputs must match the pretrained weights exactly |
| Blocks 1–4 (CoordConv) | **Keep** (frozen at first) | These hold the pretrained radar features |
| The three 1×1 convs | **Cut** | They only squash features into a 1-channel heatmap |
| GlobalMaxPool | **Cut** | The "strongest single spot wins" step your transformer replaces |

**Tap point:** output of Block 4 → **7 × 15 × 512** feature map → **105 tokens of dimension 512**.

### Your head

```
Block 4 output (7×15×512)
  → reshape to 105 tokens × 512
  → (optional) Linear 512→256 to shrink the model
  → + positional embeddings (learned 2D, or relative)
  → Transformer encoder × 2–4 layers
  → CLS token or mean pool
  → Dense(1) → logit
```

### Paper vs. code

| | Paper | Code |
|---|---|---|
| Starting filters | 32 | **64** by default (check `params.json` for overrides) |
| Convs per block | 2, 3, 3, 3 | **2, 2, 3, 3** |
| Final feature map | — | **7 × 15 × 512** |
| Output | Sigmoid → max pool | **Linear logit** → max pool (loss applied to logits) |
| Batch norm | — | None; regularization is L2 + dropout (0.1) |

### Implementation tips

- **Add a third head option.** `build_model()` already takes `head='maxpool'` or `head='mlp'`. Add `head='transformer'` as a new branch so the hybrid shares every line of input and backbone code with the baseline.
- **Grab the tap point from the pretrained model.** Find Block 4's last pool/dropout layer by name in `model.summary()`, then build `keras.Model(inputs=model.inputs, outputs=that_layer.output)`.
- **Training versions:** frozen backbone with cached features (cheap, laptop) → fine-tune → train from scratch (RunPod).

---

## 3. Core references

> ViT and ViT-hybrid come from the same paper. There is no single "CNN paper"; cite the specific ideas your model uses.

| Topic | Paper | Venue | Link |
|---|---|---|---|
| **TorNet + CNN baseline** ⭐ start here | Veillette et al., *A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data* | AMS *AIES* (2025) | [Journal](https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0006.1.xml) · [arXiv 2401.16437](https://arxiv.org/abs/2401.16437) |
| **CNN foundations** | LeCun et al., *Gradient-Based Learning Applied to Document Recognition* | Proc. IEEE (1998) | Classic citation |
| **VGG** | Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* | ICLR (2015) | [arXiv 1409.1556](https://arxiv.org/abs/1409.1556) |
| **CoordConv** | Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution* | NeurIPS (2018) | [arXiv 1807.03247](https://arxiv.org/abs/1807.03247) |
| **Transformer** | Vaswani et al., *Attention Is All You Need* | NeurIPS (2017) | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) |
| **ViT + ViT-hybrid** ⭐ direct precedent | Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* | ICLR (2021) | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) |
| **CvT** | Wu et al., *CvT: Introducing Convolutions to Vision Transformers* | ICCV (2021) | [arXiv 2103.15808](https://arxiv.org/abs/2103.15808) |

---

## 4. Related work

| Topic | Paper | Link |
|---|---|---|
| **Other TorNet models** | *KANs Layer Integration: Benchmarking Deep Learning Architectures for Tornado Prediction* (Penn State, Dec 2025). CNN, VGG19 and Xception with KAN heads; the closest analog to swapping the head. | [Penn State](https://pure.psu.edu/en/publications/kans-layer-integration-benchmarking-deep-learning-architectures-f/) |
| **Prior ML tornado detection** | Sandmæl et al., *The Tornado Probability Algorithm*, Weather and Forecasting (2023). Random forest on full-resolution polarimetric radar; cited in the TorNet paper. | Via TorNet references |

---

## 5. Optional, depending on scope

| If you… | Cite |
|---|---|
| Discuss the wider landscape | Swin ([2103.14030](https://arxiv.org/abs/2103.14030)) · CoAtNet ([2106.04803](https://arxiv.org/abs/2106.04803)) · ConvNeXt ([2201.03545](https://arxiv.org/abs/2201.03545)) |
| Use residual connections | He et al., *Deep Residual Learning for Image Recognition* ([1512.03385](https://arxiv.org/abs/1512.03385)) |
| Try pretrained ViT weights on radar features | Lu et al., *Pretrained Transformers as Universal Computation Engines* (2021) ([2103.05247](https://arxiv.org/abs/2103.05247)) |

---

## 6. Suggested reading order

1. **TorNet paper**: the dataset, the official split, and the CNN you build on.
2. **ViT paper**, especially the hybrid section: the direct precedent for your design.
3. **CvT paper**: ideas to borrow, like dropping positional embeddings for full-scan inference.
4. **KAN TorNet paper**: the closest related work on replacing the head.

---

## 7. Citation tips

- Cite the published journal or conference version when one exists; use arXiv only as a fallback.
- Pull BibTeX from Google Scholar or the publisher page so authors, years and pages are exact.
- Call your model a "CNN–transformer hybrid". Reserve "CvT" for designs that use convolutional token embeddings and projections.
- Check Google Scholar's "Cited by" on the TorNet paper before claiming novelty.

---

*Compiled October 2026. Confirm venues and years against publisher pages before final citation.*

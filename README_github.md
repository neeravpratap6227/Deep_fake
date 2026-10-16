# Deepfake Video Detection: Two-Model Ensemble

Classifies face videos as **real (0)** or **deepfake (1)** by combining two different detectors:

1. **A training-free "fact-checking" detector** based on [FACTOR](https://arxiv.org/abs/2311.01458). It never looks at a single fake. It measures how far a face is from a bank of known-real faces.
2. **A supervised CNN ensemble**, the winning solution of the [Facebook Deepfake Detection Challenge (DFDC)](https://github.com/selimsef/dfdc_deepfake_challenge). It is made of 7 EfficientNet-B7 models trained on real and fake videos.

Their scores are fused into one final decision per video. The output is `submission.csv`, which holds labels for **518 test videos**.

---

## Why two models?

| | FACTOR-style (training-free) | DFDC EfficientNet ensemble |
|---|---|---|
| Learns from fakes? | No. It only uses real faces as reference | Yes. It is trained on large deepfake datasets |
| Strength | Generalises to new or unseen fake methods | Very accurate on fake styles it has seen |
| Weakness | Weaker signal on its own | Can miss new manipulation techniques |
| Output | Distance score (higher = less like real faces) | Probability that the video is fake (0–1) |

The two approaches fail in different ways, so combining them makes the prediction more robust.

---

## Pipeline

```
 Test video (.mp4)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 ┌────────────────────┐      ┌──────────────────────────┐
 │ 1. Sample 5 frames │      │ DFDC solution            │
 │    (evenly spaced) │      │ 7 × EfficientNet-B7      │
 ├────────────────────┤      │ (pretrained weights)     │
 │ 2. Detect + align  │      └────────────┬─────────────┘
 │    face (FaceX-Zoo)│                   │  P(fake) ∈ [0,1]
 ├────────────────────┤                   │
 │ 3. 512-d face      │                   │
 │    embedding       │                   │
 ├────────────────────┤                   │
 │ 4. FAISS nearest-  │                   │
 │    neighbour dist. │                   │
 │    to REAL faces   │                   │
 ├────────────────────┤                   │
 │ 5. Mean over 5     │                   │
 │    frames          │                   │
 └─────────┬──────────┘                   │
           │  distance score              │
           └──────────────┬───────────────┘
                          ▼
         score = 0.003 × distance + 0.7 × P(fake)
                          ▼
              label = 1 (fake) if score > 0.5 else 0 (real)
```

### Step by step

**1. Frame extraction (OpenCV).** Five evenly spaced frames are taken from every video in the real reference sets (`Celeb-real`, `YouTube-real`) and the `test` set.

**2. Face detection and alignment (FaceX-Zoo).** The face closest to the centre of each frame is detected, aligned using landmarks and cropped. This removes the background so the model focuses on the face.

**3. Feature extraction.** A pretrained face-recognition backbone (`Epoch_17.pt`) turns each cropped face into a **512-dimensional embedding**.

**4. Real-face reference index (FAISS).** All embeddings from real videos go into a FAISS `IndexFlatL2` index. For each test frame, the code finds the L2 distance to its **nearest real face** (k = 1). The idea, from FACTOR, is that fake faces don't sit naturally among real ones, so they tend to land farther away.

**5. Per-video score.** The five frame distances are averaged into one `mean_column` score per video. Frames where no face was found are ignored.

**6. DFDC ensemble.** `predict_folder.py` from the DFDC winning solution runs 7 pretrained EfficientNet-B7 checkpoints on each test video and outputs a fake probability.

**7. Fusion and final label.**
```python
score = 0.003 * mean_distance + 0.7 * fake_probability
label = 1 if score > 0.5 else 0
```
Typical distances are about 50–75, so the FACTOR term adds about 0.15–0.23. That nudges borderline CNN predictions over or under the threshold.

---

## Results

| Videos in test set | Predicted real (0) | Predicted fake (1) |
|---|---|---|
| 518 | 176 | 342 |

Output format (`submission.csv`):

```csv
id,label
00011.mp4,0
00021.mp4,0
...
```

> No ground-truth labels were available for the test set, so accuracy is not reported here.

---

## Repository structure

```
├── Inference.ipynb   # Full pipeline (Colab notebook)
├── submission.csv    # Final predictions: id, label (0 = real, 1 = fake)
└── README.md
```

---

## How to run

1. Open `Inference.ipynb` in **Google Colab**.
2. Set the runtime to **GPU (T4)**: Runtime → Change runtime type.
3. Put the dataset zip (with `Celeb-real/`, `YouTube-real/`, `Celeb-Youtube-fake/`, `test/` folders) in your Google Drive and update the path in the *Download and extract dataset* cell.
4. Run all cells in order. The notebook will:
   - clone [FACTOR](https://github.com/talreiss/FACTOR), [FaceX-Zoo](https://github.com/jdhruv1503/FaceX-Zoo) and [dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
   - download the pretrained weights (face backbone and EfficientNet-B7 checkpoints)
   - write `submission.csv`

**Checkpoint cells.** Face alignment and feature extraction are slow. The notebook zips `aligned_out/` and `features/` so you can download them and re-upload them later, skipping those steps. Ignore these cells on a first run.

---

## Tech stack

`Python` · `PyTorch` · `OpenCV` · `FAISS` · `NumPy` · `pandas` · `EfficientNet-B7 (timm)` · `FaceX-Zoo` · `facenet-pytorch` · `Google Colab`

---

## Limitations and future work

- The fusion weights (0.003, 0.7) and the 0.5 threshold were **hand-tuned**. They could be learned with logistic regression on a labelled validation set.
- Only 5 frames per video are used. More frames could give a more stable score.
- If no face is detected in any frame, the FACTOR score is missing for that video.
- Adding an evaluation step (accuracy, AUC, confusion matrix) on a labelled split would quantify performance.

---

## Credits

- **FACTOR**: Reiss, Cavia & Hoshen, *Detecting Deepfakes Without Seeing Any* (2023). [arXiv:2311.01458](https://arxiv.org/abs/2311.01458)
  ```bibtex
  @article{reiss2023detecting,
    title={Detecting Deepfakes Without Seeing Any},
    author={Reiss, Tal and Cavia, Bar and Hoshen, Yedid},
    journal={arXiv preprint arXiv:2311.01458},
    year={2023}
  }
  ```
- **DFDC 1st-place solution**: Selim Seferbekov. [github.com/selimsef/dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
- **FaceX-Zoo**: face detection, alignment and recognition toolkit.
- Related reading: [DFDC dataset](https://arxiv.org/abs/1910.08854), [Face X-ray](https://arxiv.org/abs/1912.13458), [DeeperForensics-1.0](https://arxiv.org/abs/2001.03024), [CNN-generated images are surprisingly easy to spot](https://arxiv.org/abs/1912.11035).

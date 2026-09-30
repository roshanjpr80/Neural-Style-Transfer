# Restyle / StyleForge AI

**Real-time Neural Style Transfer, built two ways: entirely in the browser, and as a self-trained PyTorch model served over Flask.**

Turn any photo into any artistic style — a famous painting or one of your own — for free, with no account, and (in the browser version) without your photo ever leaving your device.

🔗 **Live demo (Flask/PyTorch version):** https://neural-style-transfer-j7vh.onrender.com

---

## Table of contents

- [What this project actually solves](#what-this-project-actually-solves)
- [Two implementations, one theory](#two-implementations-one-theory)
- [Quick start](#quick-start)
- [Getting the datasets](#getting-the-datasets)
- [Training your own decoder](#training-your-own-decoder)
- [Running the Flask app locally](#running-the-flask-app-locally)
- [Evaluating a trained model](#evaluating-a-trained-model)
- [Deployment](#deployment)
- [Tech stack](#tech-stack)
- [Known limitations](#known-limitations)
- [Theoretical background](#theoretical-background)
- [References](#references)

---

## What this project actually solves

Neural Style Transfer isn't a new algorithm — it's been solved since 2015. What's still broken for an everyday public user is **access**: most existing tools (Prisma, DeepArt.io, and similar) gate full quality behind a paywall, upload your photo to a remote server, restrict you to a fixed filter pack, or require an account just to try it once.

This project addresses those four problems directly:

| Problem in existing tools | How this project answers it                         |
| ------------------------- | --------------------------------------------------- |
| Cost / watermarking       | Free, unlimited, no watermark, no account           |
| Privacy                   | The browser version never sends your photo anywhere |
| Rigid style choice        | Any style image works, not just a curated set       |
| Friction                  | One page, no install, no sign-up                    |

## Two implementations, one theory

Both halves of this project run the same underlying method — **Adaptive Instance Normalization (AdaIN)**, from Huang & Belongie, 2017 — but solve the access problem in two different ways.

### 1. Restyle — client-side, zero backend

A single self-contained HTML file. Loads a pretrained AdaIN model via TensorFlow.js and runs every stylization directly in your browser tab. No server, no upload, no hosting cost. This is the version that best matches the project's original public-facing goal.

### 2. StyleForge AI — self-trained, Flask + PyTorch

A from-scratch PyTorch implementation of the same AdaIN architecture (`models.py`), trained on real data (`train.py`) using COCO (content) and a curated Painter by Numbers subset (style), served through a Flask web app (`app.py`, `templates/index.html`). This is the version deployed live on Render, and it's the one that demonstrates understanding of the method at the implementation level — not just using someone else's pretrained weights.

`vgg_normalised.pth`, any `decoder_*.pth` / `optimizer_*.pth` checkpoint, and the `data_set/` folder are intentionally **not** committed to this repo (see `.gitignore`) — they're either too large for git, not legally redistributable (the style dataset includes still-copyrighted artists), or both.

## Quick start

```bash
git clone https://github.com/roshanjpr80/Neural-Style-Transfer.git
cd Neural-Style-Transfer

py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

> **CPU vs GPU note:** a plain `pip install torch` on Linux resolves to a CUDA-dependent build that will fail to even `import` without a matching NVIDIA setup. If you don't have a CUDA GPU, install the CPU build first:
>
> ```bash
> pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

You'll also need `vgg_normalised.pth` — the pretrained VGG-19 encoder used as a fixed feature extractor throughout this project. Place it in the project root (or point `VGG_PATH` at wherever you keep it).

## Getting the datasets

You only need a small, curated set of images, not full bulk downloads.

**Content images (COCO):**

```bash
wget -c http://images.cocodataset.org/zips/val2017.zip   # 5,000 images, ~1GB, recommended
unzip val2017.zip
```

**Style images (Painter by Numbers, via Kaggle):**

```bash
pip install kaggle
# place your kaggle.json API token at ~/.kaggle/kaggle.json first
kaggle competitions download -c painter-by-numbers -f train_1.zip
unzip train_1.zip
```

`train_1.zip` is roughly 1/9th of the full dataset — enough variety without a 30GB+ download. Cross-reference against `train_info.csv` if you need to filter to public-domain artists for anything you plan to redistribute publicly.

## Training your own decoder

```bash
python train.py \
  --content_dir "path/to/coco_images" \
  --style_dir "path/to/style_images" \
  --vgg "vgg_normalised.pth" \
  --experiment final_run \
  --max_iter 20000 \
  --batch_size 4 \
  --lr 1e-4 \
  --style_weight 10 \
  --max_grad_norm 5.0 \
  --save_interval 2000 \
  --log_interval 50
```

- `--max_iter 20000` is a realistic target for real stylization quality without requiring GPU-days — 500–1,000 iterations only confirms the pipeline runs; below ~5,000, expect blurry/muddy output.
- `--lr 1e-4` and `--style_weight 10` match the original AdaIN paper's own hyperparameters.
- Training is iteration-driven (not epoch-based) — content and style datasets are sampled independently and endlessly via `InfiniteSampler`, so a large content set is never capped by a smaller style set.
- Checkpoints, a loss CSV, a generated loss curve, and preview images land in `experiment/<name>/`.

## Running the Flask app locally

```bash
set VGG_PATH=vgg_normalised.pth
set DECODER_PATH=experiment\final_run\decoder_final.pth
set SECRET_KEY=<generate one - see below>
python app.py
```

Generate a real secret key rather than using the insecure development fallback:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Evaluating a trained model

`code.ipynb` loads a trained checkpoint via `StyleTransferModel`, generates stylized output directly (no manual pre-made files needed), and produces two kinds of evidence:

1. **Visual** — VGG activation heatmaps at `relu1_1`–`relu4_1`, showing whether content structure survived stylization
2. **Quantitative** — cosine similarity between original and stylized features at each layer (closer to 1.0 = more content preserved)

## Deployment

The Flask app is deployed on Render. Since model weights are too large to commit to git, `render-build.sh` downloads them from an external host (Hugging Face Hub) during the build step — see the script for the exact URL format required (`/resolve/main/`, not `/blob/main/`, and the hosting repo must be public).

Build command:

```
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-deploy.txt && bash render-build.sh
```

Start command:

```
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

The standalone `neural-style-transfer.html` needs none of this — it can be hosted for free on GitHub Pages or Netlify as a single static file, since all inference happens client-side.

## Tech stack

| Layer                     | Restyle (client-side)                | StyleForge AI (server-side)   |
| ------------------------- | ------------------------------------ | ----------------------------- |
| Inference                 | TensorFlow.js, in-browser            | PyTorch, self-trained decoder |
| Interface                 | Vanilla HTML/CSS/JS                  | Flask + Jinja2 + Bootstrap 5  |
| Hosting                   | Static file (GitHub Pages / Netlify) | Render (gunicorn)             |
| Where computation happens | User's device                        | Server                        |

## Known limitations

- **CPU inference is slow**, and the deployed Render instance has no GPU — expect real latency on the live demo, especially on the free tier.
- **A decoder checkpoint is only as good as its training.** A checkpoint saved early in training (e.g. a few hundred iterations) will produce blurry, undertrained output — check how far a checkpoint was actually trained before treating its results as representative.
- Free-tier Render spins down on inactivity; the first request after idling can take 30–60 seconds.
- Uploaded/generated images on the Flask app don't persist across redeploys (ephemeral disk) unless a paid persistent disk is attached.

## Theoretical background

Full write-up — problem definition, content vs. style, why NST is hard, evaluation methodology, and a survey of alternative NST method families (fixed-style feed-forward, WCT, GAN-based, diffusion-based) — is in the accompanying technical documentation. In short, this project's architecture (AdaIN) is the only method family that simultaneously satisfies: arbitrary style at inference time, real-time speed, and being small enough to run entirely client-side — which is exactly what the public-facing problem statement requires.

## References

- Simonyan, K., & Zisserman, A. (2015). _Very Deep Convolutional Networks for Large-Scale Image Recognition._ ICLR. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). _A Neural Algorithm of Artistic Style._ [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). _Perceptual Losses for Real-Time Style Transfer and Super-Resolution._ ECCV. [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)
- Huang, X., & Belongie, S. (2017). _Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization._ ICCV. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)

---

_An academic project exploring how a technically serious deep learning capability can be delivered to the public in a form that is simultaneously free, private, flexible, and immediate._

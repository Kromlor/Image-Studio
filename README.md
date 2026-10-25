# Image Studio

Riley's private, offline AI image generator for Windows (NVIDIA GPU).
Runs FLUX.1 Schnell locally and can train custom character models (LoRA).

This repo holds **only the app code**. Pictures, models, training data and the
~32 GB engine live on the PC and are excluded by `.gitignore`.

## Set up on a new PC
1. Download this repo (Code → Download ZIP) and unzip it to the drive you want (e.g. `G:\Image Studio`).
2. Double-click **Install Image Studio** (about 35 GB download; needs a free Hugging Face token once).
3. Double-click **Start Image Studio**.

See `READ ME FIRST.txt` for everyday use.

Engine: FLUX.1 Schnell (Black Forest Labs, Apache 2.0). Training add-on: ostris/FLUX.1-schnell-training-adapter (Apache 2.0).

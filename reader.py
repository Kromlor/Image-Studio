"""
Image Studio "picture reader": describes ONLY the clothing in an uploaded picture.
Runs as its own process so its graphics memory is fully freed when it finishes.
Model: Qwen2.5-VL 7B Instruct (Apache 2.0), 4-bit.
Usage: reader.py --image IN.png --out result.json
"""
import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

READER_DIR = MODELS_DIR / "outfit-reader"            # ready-to-use 4-bit copy lives here
FULL_DIR = MODELS_DIR / "outfit-reader-download"     # temporary full download (fallback path only)
PACKED_REPO = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
OFFICIAL_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
FAKE = os.environ.get("IMAGE_STUDIO_FAKE") == "1"

PROMPT = (
    "Describe ONLY the clothing and accessories in this picture so an artist could recreate the outfit exactly: "
    "each garment, its color, material, pattern, cut and fit, plus shoes, jewelry, bags and other accessories. "
    "Do NOT describe the person, their body, skin, face, hair, age, pose, expression, or the background. "
    "Write one compact comma-separated description under 60 words, starting with the main garment. "
    "If there is no clothing in the picture, reply exactly: NO OUTFIT"
)


def status(path: Path, **kw):
    if path:
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(kw), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass


def ensure_model(stat):
    if (READER_DIR / "config.json").exists():
        return
    from huggingface_hub import snapshot_download
    try:
        stat(phase="download", message="One time: downloading the picture reader (about 6 GB)…")
        snapshot_download(PACKED_REPO, local_dir=str(READER_DIR))
        if (READER_DIR / "config.json").exists():
            return
    except Exception:
        print("Packed reader download failed, trying the official one:\n" + traceback.format_exc(), flush=True)
        shutil.rmtree(READER_DIR, ignore_errors=True)

    # Fallback: official weights, pack to 4-bit ourselves, keep only the packed copy.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    stat(phase="download", message="One time: downloading the picture reader (about 17 GB, then packed to 6 GB)…")
    snapshot_download(OFFICIAL_REPO, local_dir=str(FULL_DIR))
    stat(phase="pack", message="One time: packing the picture reader…")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(str(FULL_DIR), quantization_config=q,
                                                        dtype=torch.bfloat16, device_map="cuda:0")
    model.save_pretrained(str(READER_DIR))
    AutoProcessor.from_pretrained(str(FULL_DIR)).save_pretrained(str(READER_DIR))
    del model
    shutil.rmtree(FULL_DIR, ignore_errors=True)


def read_outfit(image_path: Path, stat) -> str:
    if FAKE:
        import time
        time.sleep(1.0)
        return "cropped denim jacket with silver buttons, white ribbed tank top, high-waisted black jeans, white canvas sneakers, thin gold necklace"
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    ensure_model(stat)
    stat(phase="load", message="Loading the picture reader…")
    processor = AutoProcessor.from_pretrained(str(READER_DIR), min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    model = AutoModelForImageTextToText.from_pretrained(str(READER_DIR), dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    stat(phase="read", message="Reading the outfit…")
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return " ".join(text.split()).strip().strip('"').rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", default="")
    a = ap.parse_args()
    out = Path(a.out)
    st = Path(a.status) if a.status else None
    stat = lambda **kw: status(st, **kw)
    try:
        text = read_outfit(Path(a.image), stat)
        if text.upper().startswith("NO OUTFIT") or not text:
            out.write_text(json.dumps({"ok": False, "error": "I couldn't find any clothing in that picture."}), encoding="utf-8")
        else:
            out.write_text(json.dumps({"ok": True, "text": text}), encoding="utf-8")
        return 0
    except Exception as e:
        print("ERROR:\n" + traceback.format_exc(), flush=True)
        msg = str(e)
        if "out of memory" in msg.lower():
            err = "The graphics card ran out of memory. Close games or other heavy apps and try again."
        elif any(x in msg for x in ("401", "403", "gated", "Gated")):
            err = "The picture reader download needs a Hugging Face sign-in. Tell Claude and it'll walk you through it."
        elif any(x in msg.lower() for x in ("connection", "resolve", "timed out", "offline")):
            err = "The one-time picture reader download needs the internet. Check your connection and try again."
        else:
            err = "The picture reader hit an error. Send reader_log.txt (in the Image Studio folder) to Claude."
        out.write_text(json.dumps({"ok": False, "error": err}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())

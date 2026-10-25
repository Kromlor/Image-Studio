"""
Image Studio trainer: teaches FLUX.1 Schnell a character or style (a LoRA "model").

How it works (QLoRA on a 16 GB card):
  1. Reads every caption once with the text readers, then unloads them.
  2. Turns every training picture into the engine's internal format once (VAE latents).
  3. Loads the packed 4-bit engine, switches on the Schnell training add-on
     (ostris/FLUX.1-schnell-training-adapter) so training doesn't break Schnell's 4-step speed,
     and trains a small set of new weights on top.
  4. Every quarter of the way it makes sample pictures WITHOUT the add-on (how you'll use it).
  5. Saves only your new weights to "My Models".
Progress goes to progress.json; the app writes "stop" or "keep" to control.txt to end early.
"""
import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE = MODELS_DIR / "flux-schnell"
PACKED_T = MODELS_DIR / "flux-schnell-transformer-nf4"
PACKED_TE2 = MODELS_DIR / "flux-schnell-te2-nf4"
ADAPTER_DIR = MODELS_DIR / "schnell-training-adapter"
ADAPTER_REPO = "ostris/FLUX.1-schnell-training-adapter"
FAKE = os.environ.get("IMAGE_STUDIO_FAKE") == "1"

RES = 768          # training size (longest side)
RANK = 16
LR = 1e-4
SAMPLE_RES = 768
TARGETS = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
           "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"]


def say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Progress:
    def __init__(self, path: Path, total: int):
        self.path = path
        self.d = {"status": "starting", "message": "Getting ready…", "step": 0, "total": total,
                  "eta_s": None, "loss": None, "samples": [], "started": time.time()}
        self._last = 0.0
        self.write()

    def set(self, force=True, **kw):
        self.d.update(kw)
        now = time.time()
        if force or now - self._last > 1.5:
            self.write()
            self._last = now

    def write(self):
        tmp = self.path.with_suffix(".tmp")
        for _ in range(10):
            try:
                tmp.write_text(json.dumps(self.d), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(0.1)


def control(proj: Path) -> str:
    try:
        return (proj / "control.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def fit_size(w, h, res=RES):
    s = res / max(w, h)
    return max(256, int(w * s) // 16 * 16), max(256, int(h * s) // 16 * 16)


def save_meta(out: Path, meta: dict, steps: int):
    out.with_suffix(".json").write_text(json.dumps({
        "name": meta.get("name", out.stem), "word": meta.get("word", ""), "steps": steps,
        "base": "FLUX.1-schnell", "rank": RANK,
        "created": datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- practice mode
def fake_run(proj, meta, steps, out, prog):
    from PIL import Image
    first = sorted((proj / "images").glob("*.png"))[0]
    steps = min(steps, 40)
    prog.set(total=steps, status="running", message="Learning your pictures…")
    t0 = time.time()
    for step in range(1, steps + 1):
        c = control(proj)
        if c == "stop":
            prog.set(status="cancelled", message="Training stopped. Nothing was saved.")
            return 0
        time.sleep(0.15)
        el = time.time() - t0
        prog.set(force=False, step=step, loss=round(0.5 / step ** 0.3, 4), eta_s=int(el / step * (steps - step)))
        if step % 10 == 0 or c == "keep":
            names = []
            for k in range(2):
                n = f"step_{step:05d}_{k}.png"
                Image.open(first).convert("RGB").resize((384, 384)).save(proj / "samples" / n)
                names.append(n)
            prog.d["samples"].append({"step": step, "files": names})
            prog.set()
        if c == "keep":
            break
    out.write_bytes(b"practice")
    save_meta(out, meta, step)
    prog.set(status="done", step=step, eta_s=0, message="Done! Your model is ready.", model=out.stem)
    return 0


# ---------------------------------------------------------------- real training
def get_adapter(prog) -> Path:
    files = sorted(ADAPTER_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
    if files:
        return files[0]
    prog.set(status="starting", message="One time: downloading the Schnell training add-on…")
    from huggingface_hub import snapshot_download
    snapshot_download(ADAPTER_REPO, local_dir=str(ADAPTER_DIR), allow_patterns=["*.safetensors", "*.json", "README.md"])
    files = sorted(ADAPTER_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Training add-on download had no weights file.")
    return files[0]


def load_te2(torch):
    from transformers import T5EncoderModel, BitsAndBytesConfig as TBnb

    def load(path, **kw):
        try:
            return T5EncoderModel.from_pretrained(path, dtype=torch.bfloat16, **kw)
        except TypeError:
            return T5EncoderModel.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)
    if (PACKED_TE2 / "config.json").exists():
        return load(str(PACKED_TE2))
    q = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    return load(str(BASE / "text_encoder_2"), quantization_config=q)


def load_transformer(torch):
    from diffusers import FluxTransformer2DModel, BitsAndBytesConfig
    if (PACKED_T / "config.json").exists():
        return FluxTransformer2DModel.from_pretrained(str(PACKED_T), torch_dtype=torch.bfloat16)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    return FluxTransformer2DModel.from_pretrained(str(BASE), subfolder="transformer",
                                                  quantization_config=q, torch_dtype=torch.bfloat16)


def real_run(proj, meta, steps, out, prog):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import FluxPipeline, AutoencoderKL

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    dev, dt = "cuda", torch.bfloat16
    word = meta["word"]
    caps = meta.get("captions", {})
    files = sorted((proj / "images").glob("*.png"))
    say(f"{len(files)} images, {steps} steps, word '{word}'")

    adapter = get_adapter(prog)
    say(f"Training add-on: {adapter.name}")

    # 1) captions -> embeddings (text readers loaded once, then freed)
    prog.set(status="starting", message="Reading your captions…")
    te2 = load_te2(torch)
    tpipe = FluxPipeline.from_pretrained(str(BASE), transformer=None, vae=None, text_encoder_2=te2, torch_dtype=dt)
    tpipe.text_encoder.to(dev)

    def embed(text):
        with torch.no_grad():
            pe, pooled, _ = tpipe.encode_prompt(prompt=text, prompt_2=None, device=dev,
                                                num_images_per_prompt=1, max_sequence_length=256)
        return pe.to("cpu", dt), pooled.to("cpu", dt)

    texts = [f"{word}, {caps.get(f.name, '').strip()}".rstrip(", ") for f in files]
    embeds = [embed(t) for t in texts]
    sample_prompts = [f"{word}, full body, standing in a sunlit forest clearing",
                      f"{word}, portrait, head and shoulders, soft studio light"]
    sample_embeds = [embed(t) for t in sample_prompts]
    del tpipe, te2
    gc.collect(); torch.cuda.empty_cache()

    # 2) pictures -> latents
    prog.set(message="Preparing your pictures…")
    vae = AutoencoderKL.from_pretrained(str(BASE), subfolder="vae", torch_dtype=dt).to(dev)
    vae.requires_grad_(False)
    latents = []
    for f in files:
        img = Image.open(f).convert("RGB")
        w, h = fit_size(*img.size)
        img = img.resize((w, h), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img).astype("float32") / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev, dt)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample()
        lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
        latents.append(lat.to("cpu", dt))
    say("Latents cached: " + ", ".join(f"{tuple(l.shape[2:])}" for l in latents[:5]) + (" …" if len(latents) > 5 else ""))

    # 3) engine + add-on + your new weights
    prog.set(message="Loading the engine for training…")
    transformer = load_transformer(torch)
    transformer.requires_grad_(False)
    pipe = FluxPipeline.from_pretrained(str(BASE), transformer=transformer, vae=vae, text_encoder=None,
                                        text_encoder_2=None, tokenizer=None, tokenizer_2=None, torch_dtype=dt)
    pipe.load_lora_weights(str(adapter), adapter_name="assistant")

    from peft import LoraConfig
    transformer.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian",
                                       target_modules=TARGETS), adapter_name="user")

    def train_mode():
        transformer.set_adapters(["assistant", "user"], weights=[1.0, 1.0])
        params = []
        for n, p in transformer.named_parameters():
            mine = "lora_" in n and ".user." in n
            p.requires_grad_(mine)
            if mine:
                if p.dtype != torch.float32:
                    p.data = p.data.float()
                params.append(p)
        transformer.train()
        return params

    params = train_mode()
    if not params:
        raise RuntimeError("No trainable weights were created.")
    say(f"Trainable weights: {sum(p.numel() for p in params) / 1e6:.1f}M")
    transformer.enable_gradient_checkpointing()
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=LR, weight_decay=1e-4)
    except Exception:
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    def sample(step):
        prog.set(status="sampling", message=f"Making sample pictures at step {step}…")
        transformer.eval()
        transformer.set_adapters(["user"], weights=[1.0])        # add-on OFF: exactly how you'll use it
        names = []
        with torch.no_grad():
            for k, (pe, pooled) in enumerate(sample_embeds):
                g = torch.Generator("cpu").manual_seed(1234 + k)
                img = pipe(prompt_embeds=pe.to(dev), pooled_prompt_embeds=pooled.to(dev),
                           width=SAMPLE_RES, height=SAMPLE_RES, num_inference_steps=4,
                           guidance_scale=0.0, generator=g).images[0]
                n = f"step_{step:05d}_{k}.png"
                img.save(proj / "samples" / n)
                names.append(n)
        prog.d["samples"].append({"step": step, "files": names})
        return train_mode()

    def save(step):
        from peft.utils import get_peft_model_state_dict
        prog.set(status="saving", message="Saving your model…")
        sd = get_peft_model_state_dict(transformer, adapter_name="user")
        sd = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in sd.items()}
        tmpdir = out.parent / f".{out.stem}_tmp"
        tmpdir.mkdir(exist_ok=True)
        FluxPipeline.save_lora_weights(save_directory=str(tmpdir), transformer_lora_layers=sd,
                                       weight_name="lora.safetensors")
        os.replace(tmpdir / "lora.safetensors", out)
        try:
            tmpdir.rmdir()
        except OSError:
            pass
        save_meta(out, meta, step)
        say(f"Saved {out.name} ({len(sd)} tensors)")

    # 4) train
    quarters = {max(1, int(steps * f)) for f in (0.25, 0.5, 0.75)}
    prog.set(status="running", message="Learning your pictures…")
    t0, train_time, loss_avg = time.time(), 0.0, None
    step = 0
    for step in range(1, steps + 1):
        c = control(proj)
        if c == "stop":
            prog.set(status="cancelled", message="Training stopped. Nothing was saved.")
            return 0
        if c == "keep":
            step -= 1
            break
        ts = time.time()
        i = random.randrange(len(latents))
        x0 = latents[i].to(dev, dt)
        pe, pooled = embeds[i]
        b, ch, h, w = x0.shape
        noise = torch.randn_like(x0)
        sigma = torch.sigmoid(torch.randn((b,), device=dev, dtype=torch.float32))
        s4 = sigma.view(b, 1, 1, 1).to(dt)
        noisy = (1 - s4) * x0 + s4 * noise
        packed = FluxPipeline._pack_latents(noisy, b, ch, h, w)
        target = FluxPipeline._pack_latents(noise - x0, b, ch, h, w)
        img_ids = FluxPipeline._prepare_latent_image_ids(b, h // 2, w // 2, dev, dt)
        txt_ids = torch.zeros(pe.shape[1], 3, device=dev, dtype=dt)
        with torch.autocast("cuda", dtype=dt):
            pred = transformer(hidden_states=packed, timestep=sigma.to(dt), guidance=None,
                               pooled_projections=pooled.to(dev), encoder_hidden_states=pe.to(dev),
                               txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        lv = loss.item()
        loss_avg = lv if loss_avg is None else loss_avg * 0.95 + lv * 0.05
        train_time += time.time() - ts
        eta = int(train_time / step * (steps - step) + 45 * len([q for q in quarters if q > step]))
        prog.set(force=False, status="running", message="Learning your pictures…", step=step,
                 loss=round(loss_avg, 4), eta_s=eta)
        if step == 1 or step % 50 == 0:
            say(f"step {step}/{steps} loss {loss_avg:.4f} {train_time / step:.2f}s/step "
                f"vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        if step in quarters:
            params = sample(step)
            opt.param_groups[0]["params"] = params

    save(step)
    sample(step)
    prog.set(status="done", step=step, eta_s=0, message="Done! Your model is ready.", model=out.stem)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proj, out = Path(a.project), Path(a.out)
    meta = json.loads((proj / "project.json").read_text(encoding="utf-8"))
    (proj / "samples").mkdir(exist_ok=True)
    prog = Progress(proj / "progress.json", a.steps)
    try:
        return (fake_run if FAKE else real_run)(proj, meta, a.steps, out, prog)
    except Exception as e:
        say("ERROR:\n" + traceback.format_exc())
        msg = str(e)
        if "out of memory" in msg.lower():
            text = "The graphics card ran out of memory. Close games and other heavy apps, then try again."
        elif any(x in msg for x in ("401", "403", "gated", "Gated")):
            text = "The training add-on download needs a Hugging Face sign-in. Tell Claude and it'll walk you through it."
        else:
            text = "Training stopped because of an error. Send train_log.txt from the project folder to Claude."
        prog.set(status="error", message=text)
        return 1


if __name__ == "__main__":
    sys.exit(main())

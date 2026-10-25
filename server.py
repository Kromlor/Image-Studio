"""
Riley's Image Studio v2.1
A private image generator that runs only on this PC (address 127.0.0.1 = this computer only).
Engine: FLUX.1 Schnell (open weights, Apache-2.0). Trains your own character models (LoRA).
"""
import os
import sys
import gc
import json
import time
import base64
import io
import random
import re
import shutil
import socket
import subprocess
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

VERSION = "2.1"
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
IMAGES_DIR = APP_DIR / "My Images"
TRAIN_DIR = APP_DIR / "Training"
MYMODELS_DIR = APP_DIR / "My Models"
HISTORY_FILE = IMAGES_DIR / "history.jsonl"
OUTFITS_DIR = APP_DIR / "Outfits"
SCENES_DIR = APP_DIR / "Scenes"
CHARS_FILE = APP_DIR / "Characters" / "characters.json"
LOG_FILE = APP_DIR / "studio_log.txt"
BASE_MODEL = MODELS_DIR / "flux-schnell"
PACKED_TRANSFORMER = MODELS_DIR / "flux-schnell-transformer-nf4"
PACKED_TE2 = MODELS_DIR / "flux-schnell-te2-nf4"

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ["HF_HUB_OFFLINE"] = "1"          # generating never uses the internet
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FAKE = os.environ.get("IMAGE_STUDIO_FAKE") == "1"   # practice mode for testing without a GPU
PORT = int(os.environ.get("IMAGE_STUDIO_PORT", "7860"))
HOST = "127.0.0.1"

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, PngImagePlugin

NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0   # child processes never pop up a console
for d in (IMAGES_DIR, TRAIN_DIR, MYMODELS_DIR, OUTFITS_DIR, SCENES_DIR, CHARS_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


STATE = {"status": "loading", "message": "Warming up the engine…", "mode": ""}
PIPE = None
PIPE_MODE = ""            # "gpu" (everything on the card) or "offload"
LOADED_LORAS: set = set()
GEN_LOCK = threading.RLock()
FILE_LOCK = threading.Lock()
TRAIN = {"proc": None, "slug": None}
INPAINT = {"pipe": None, "src": None}

SHAPES = {"square": (1024, 1024), "tall": (896, 1152), "wide": (1152, 896)}
STYLES = {
    "none": "",
    "illustration": ", detailed storybook illustration, clean linework, rich color",
    "watercolor": ", soft watercolor painting, textured paper, gentle washes of color",
    "photo": ", realistic photograph, natural lighting, sharp focus, high detail",
    "fantasy": ", epic fantasy art, dramatic lighting, painterly, highly detailed",
    "pixel": ", pixel art, 16-bit video game style, crisp pixels, limited palette",
    "comic": ", bold comic book art, ink outlines, flat vivid colors",
}
LENGTHS = {"quick": 600, "standard": 1000, "thorough": 1600}


# ================================================================ engine
def _load_te2(torch):
    """T5 text encoder packed to 4-bit so the whole engine fits on a 16 GB card."""
    from transformers import T5EncoderModel, BitsAndBytesConfig as TBnb

    def load(path, **kw):
        try:
            return T5EncoderModel.from_pretrained(path, dtype=torch.bfloat16, **kw)
        except TypeError:
            return T5EncoderModel.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)

    if (PACKED_TE2 / "config.json").exists():
        return load(str(PACKED_TE2))
    STATE["message"] = "One time: packing the text reader (a minute or two)…"
    q = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    te2 = load(str(BASE_MODEL / "text_encoder_2"), quantization_config=q)
    try:
        te2.save_pretrained(str(PACKED_TE2))
        log("Packed text reader saved.")
    except Exception:
        log("Could not save packed text reader:\n" + traceback.format_exc())
    return te2


def _load_transformer(torch):
    from diffusers import FluxTransformer2DModel
    if (PACKED_TRANSFORMER / "config.json").exists():
        return FluxTransformer2DModel.from_pretrained(str(PACKED_TRANSFORMER), torch_dtype=torch.bfloat16)
    STATE["message"] = "One time: packing the engine to fit your card (a few minutes)…"
    from diffusers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(str(BASE_MODEL), subfolder="transformer",
                                                quantization_config=q, torch_dtype=torch.bfloat16)
    try:
        tr.save_pretrained(str(PACKED_TRANSFORMER))
        log("Packed engine saved.")
    except Exception:
        log("Could not save packed engine:\n" + traceback.format_exc())
    return tr


def load_engine() -> None:
    global PIPE, PIPE_MODE
    try:
        STATE.update(status="loading", message="Warming up the engine…", mode="")
        if FAKE:
            time.sleep(1.0)
            PIPE_MODE = "gpu"
            STATE.update(status="ready", message="Practice mode (no real engine)", mode="practice")
            log("Practice mode ready.")
            return

        import torch
        if not torch.cuda.is_available():
            STATE.update(status="error",
                         message="Your NVIDIA graphics card isn't visible to the engine. "
                                 "Update your NVIDIA driver, restart, and try again.")
            return
        log(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
        if not (BASE_MODEL / "model_index.json").exists():
            STATE.update(status="error",
                         message="The picture engine files are missing. Double-click “Install Image Studio” first.")
            return

        from diffusers import FluxPipeline
        STATE["message"] = "Loading the engine…"
        transformer = None
        try:
            transformer = _load_transformer(torch)
        except Exception:
            log("Packed engine unavailable, using safe mode:\n" + traceback.format_exc())

        if transformer is None:
            PIPE = FluxPipeline.from_pretrained(str(BASE_MODEL), torch_dtype=torch.bfloat16)
            PIPE.enable_sequential_cpu_offload()
            PIPE_MODE = "offload"
            STATE.update(status="ready", message="Ready", mode="safe (slower)")
            log("Engine ready (safe mode).")
            return

        te2 = None
        try:
            te2 = _load_te2(torch)
        except Exception:
            log("Packed text reader unavailable:\n" + traceback.format_exc())

        STATE["message"] = "Loading the engine…"
        kwargs = {"transformer": transformer}
        if te2 is not None:
            kwargs["text_encoder_2"] = te2
        PIPE = FluxPipeline.from_pretrained(str(BASE_MODEL), torch_dtype=torch.bfloat16, **kwargs)
        if te2 is not None:
            try:
                PIPE.to("cuda")
                PIPE_MODE = "gpu"
            except Exception:
                log("Could not keep everything on the card:\n" + traceback.format_exc())
                PIPE.enable_model_cpu_offload()
                PIPE_MODE = "offload"
        else:
            PIPE.enable_model_cpu_offload()
            PIPE_MODE = "offload"
        LOADED_LORAS.clear()
        STATE.update(status="ready", message="Ready", mode="fast" if PIPE_MODE == "gpu" else "standard")
        log(f"Engine ready ({PIPE_MODE}).")
    except Exception:
        log("Engine failed to start:\n" + traceback.format_exc())
        STATE.update(status="error",
                     message="The engine couldn't start. Send the file studio_log.txt to Claude to fix it.")


def unload_engine() -> None:
    global PIPE
    with GEN_LOCK:
        PIPE = None
        INPAINT.update(pipe=None, src=None)
        LOADED_LORAS.clear()
        gc.collect()
        if not FAKE:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


def lora_name(slug: str) -> str:
    return "m_" + re.sub(r"[^a-zA-Z0-9]", "_", slug)


def apply_model(slug: str, strength: float) -> None:
    """Switch the engine between the standard look and one of your trained models."""
    if FAKE or PIPE is None:
        return
    if not slug:
        if LOADED_LORAS:
            PIPE.disable_lora()
        return
    name = lora_name(slug)
    if name not in LOADED_LORAS:
        PIPE.load_lora_weights(str(MYMODELS_DIR / f"{slug}.safetensors"), adapter_name=name)
        LOADED_LORAS.add(name)
    PIPE.enable_lora()
    PIPE.set_adapters([name], adapter_weights=[strength])


def fake_image(prompt: str, w: int, h: int, seed: int) -> Image.Image:
    rnd = random.Random(seed)
    a = (rnd.randint(40, 200), rnd.randint(40, 200), rnd.randint(40, 200))
    b = (rnd.randint(40, 200), rnd.randint(40, 200), rnd.randint(40, 200))
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3)))
    d.text((40, 40), "PRACTICE MODE\n" + prompt[:80], fill=(255, 255, 255))
    time.sleep(0.6)
    return img


# ================================================================ helpers
def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-")) or "item"


def safe_name(name: str, ext: str = "png") -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+\." + ext, name or ""):
        raise HTTPException(404, "Not found")
    return name


def to_recycle_bin(path: Path) -> bool:
    """Send a file or folder to the Windows Recycle Bin. Returns False if that isn't possible."""
    if not sys.platform.startswith("win") or not path.exists():
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                        ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
        FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
        op = SHFILEOPSTRUCTW(None, FO_DELETE, str(path.resolve()) + "\0", None,
                             FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI, False, None, None)
        return ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op)) == 0 and not path.exists()
    except Exception:
        log("Recycle Bin failed:\n" + traceback.format_exc())
        return False


def remove_safely(path: Path, fallback_dir: Path) -> None:
    """Recycle Bin first; if that's unavailable, move into a _Deleted folder (still recoverable)."""
    if to_recycle_bin(path):
        return
    fallback_dir.mkdir(parents=True, exist_ok=True)
    dest = fallback_dir / path.name
    if dest.exists():
        dest = fallback_dir / f"{path.stem}_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(dest))


def read_history() -> list:
    items = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (IMAGES_DIR / rec.get("file", "")).exists():
                    items.append(rec)
    items.reverse()
    return items


def decode_data_url(data: str) -> Image.Image:
    m = re.match(r"data:image/[a-zA-Z+.-]+;base64,(.*)", data or "", re.S)
    if not m:
        raise HTTPException(400, "That file isn't a picture.")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(m.group(1))))
        img.load()
    except Exception:
        raise HTTPException(400, "That picture couldn't be opened.")
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("RGB")


def read_json(path: Path, default):
    for _ in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.05)
    return default


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def project_dir(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", slug or ""):
        raise HTTPException(404, "Project not found")
    p = TRAIN_DIR / slug
    if not (p / "project.json").exists():
        raise HTTPException(404, "Project not found")
    return p


def list_models() -> list:
    out = []
    for f in sorted(MYMODELS_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = read_json(f.with_suffix(".json"), {})
        out.append({"slug": f.stem, "name": meta.get("name", f.stem), "word": meta.get("word", ""),
                    "created": meta.get("created", ""), "steps": meta.get("steps")})
    return out


# ================================================================ web app
app = FastAPI(title="Image Studio", docs_url=None, redoc_url=None, openapi_url=None)


class GenRequest(BaseModel):
    prompt: str
    style: str = "none"
    shape: str = "square"
    seed: int | None = None
    model: str = ""
    strength: float = 1.0
    outfit: str = ""
    outfit_name: str = ""


def outfit_clause(text: str) -> str:
    t = " ".join((text or "").split())[:600].strip(" ,.")
    return f", wearing {t}" if t else ""


@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return {**STATE, "version": VERSION, "fake": FAKE, "training": TRAIN["slug"]}


@app.post("/api/generate")
def generate(req: GenRequest):
    if STATE["status"] != "ready":
        raise HTTPException(503, STATE["message"])
    prompt = " ".join((req.prompt or "").split())[:1500]
    if not prompt:
        raise HTTPException(400, "Type an idea first.")
    style = req.style if req.style in STYLES else "none"
    shape = req.shape if req.shape in SHAPES else "square"
    w, h = SHAPES[shape]
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    model, model_name, word = "", "", ""
    if req.model:
        meta_path = MYMODELS_DIR / f"{req.model}.json"
        if not re.fullmatch(r"[a-z0-9-]+", req.model) or not (MYMODELS_DIR / f"{req.model}.safetensors").exists():
            raise HTTPException(400, "That trained model wasn't found.")
        meta = read_json(meta_path, {})
        model, model_name, word = req.model, meta.get("name", req.model), meta.get("word", "")
    strength = max(0.0, min(1.5, float(req.strength)))
    run_prompt = prompt
    if word and word.lower() not in prompt.lower():
        run_prompt = f"{word}, {prompt}"           # the code word is what calls up your character
    full_prompt = run_prompt + outfit_clause(req.outfit) + STYLES[style]

    with GEN_LOCK:
        if PIPE is None and not FAKE:
            raise HTTPException(503, STATE["message"])
        t0 = time.time()
        try:
            if FAKE:
                img = fake_image(("[" + model_name + "] " if model else "") + full_prompt, w, h, seed)
            else:
                import torch
                apply_model(model, strength)
                gen = torch.Generator("cpu").manual_seed(seed)
                img = run_pipe(full_prompt, w, h, gen)
        except Exception as e:
            log("Generation failed:\n" + traceback.format_exc())
            if "out of memory" in str(e).lower():
                raise HTTPException(500, "The graphics card ran out of memory. Close games or other heavy apps and try again.")
            raise HTTPException(500, "Something went wrong making that picture. Try again; if it keeps happening, send studio_log.txt to Claude.")
        secs = round(time.time() - t0, 1)

    name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{slugify(prompt)}_{seed}.png"
    meta = PngImagePlugin.PngInfo()
    for k, v in (("prompt", run_prompt), ("style", style), ("seed", str(seed)),
                 ("engine", "FLUX.1-schnell"), ("model", model_name or "Standard")):
        meta.add_text(k, v)
    img.save(IMAGES_DIR / name, pnginfo=meta)
    rec = {"file": name, "prompt": prompt, "style": style, "shape": shape, "seed": seed,
           "seconds": secs, "created": datetime.now().isoformat(timespec="seconds"),
           "model": model, "model_name": model_name, "strength": strength if model else None,
           "outfit": " ".join((req.outfit or "").split())[:600], "outfit_name": req.outfit_name[:60]}
    add_history(rec)
    log(f"Made {name} in {secs}s" + (f" with {model_name}" if model else ""))
    return rec


def add_history(rec: dict) -> None:
    with FILE_LOCK, open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_pipe(prompt, w, h, gen):
    """Generate; if the card runs out of memory in all-on-card mode, switch to standard mode once and retry."""
    global PIPE_MODE
    import torch
    try:
        return PIPE(prompt, width=w, height=h, num_inference_steps=4, guidance_scale=0.0,
                    max_sequence_length=256, generator=gen).images[0]
    except torch.cuda.OutOfMemoryError:
        if PIPE_MODE != "gpu":
            raise
        log("Out of memory with everything on the card; switching to standard mode.")
        torch.cuda.empty_cache()
        PIPE.enable_model_cpu_offload()
        PIPE_MODE = "offload"
        STATE["mode"] = "standard"
        return PIPE(prompt, width=w, height=h, num_inference_steps=4, guidance_scale=0.0,
                    max_sequence_length=256, generator=gen).images[0]


@app.get("/api/gallery")
def gallery():
    return read_history()


@app.get("/images/{name}")
def image(name: str):
    p = IMAGES_DIR / safe_name(name)
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")


class FilesReq(BaseModel):
    files: list[str]


@app.post("/api/delete-images")
def delete_images(req: FilesReq):
    names = {safe_name(n) for n in req.files}
    removed = 0
    with FILE_LOCK:
        for n in names:
            p = IMAGES_DIR / n
            if p.exists():
                remove_safely(p, IMAGES_DIR / "_Deleted")
                removed += 1
        if HISTORY_FILE.exists():
            lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
            keep = []
            for line in lines:
                try:
                    if json.loads(line).get("file") in names:
                        continue
                except json.JSONDecodeError:
                    pass
                keep.append(line)
            tmp = HISTORY_FILE.with_suffix(".tmp")
            tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            os.replace(tmp, HISTORY_FILE)
    log(f"Moved {removed} picture(s) to the Recycle Bin.")
    return {"removed": removed}


@app.post("/api/open-folder")
def open_folder(which: str = "images"):
    target = {"images": IMAGES_DIR, "models": MYMODELS_DIR, "training": TRAIN_DIR}.get(which, IMAGES_DIR)
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # noqa: Windows only
        return {"ok": True}
    return {"ok": False, "path": str(target)}


# ================================================================ my models
@app.get("/api/models")
def models():
    return list_models()


class RenameReq(BaseModel):
    name: str


@app.post("/api/models/{slug}/rename")
def rename_model(slug: str, req: RenameReq):
    f = MYMODELS_DIR / f"{slug}.safetensors"
    if not re.fullmatch(r"[a-z0-9-]+", slug) or not f.exists():
        raise HTTPException(404, "Model not found")
    meta = read_json(f.with_suffix(".json"), {})
    meta["name"] = " ".join(req.name.split())[:60] or meta.get("name", slug)
    write_json(f.with_suffix(".json"), meta)
    return {"ok": True}


@app.post("/api/models/{slug}/delete")
def delete_model(slug: str):
    f = MYMODELS_DIR / f"{slug}.safetensors"
    if not re.fullmatch(r"[a-z0-9-]+", slug) or not f.exists():
        raise HTTPException(404, "Model not found")
    with GEN_LOCK:
        name = lora_name(slug)
        if name in LOADED_LORAS and PIPE is not None:
            try:
                PIPE.delete_adapters(name)
            except Exception:
                log("Could not unload model:\n" + traceback.format_exc())
            LOADED_LORAS.discard(name)
        for p in (f, f.with_suffix(".json")):
            if p.exists():
                remove_safely(p, MYMODELS_DIR / "_Deleted")
    log(f"Model {slug} moved to the Recycle Bin.")
    return {"ok": True}


# ================================================================ training projects
class NewProject(BaseModel):
    name: str
    word: str


@app.get("/api/train/projects")
def projects():
    out = []
    for p in sorted(TRAIN_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta = read_json(p / "project.json", None) if p.is_dir() else None
        if not meta:
            continue
        out.append({"slug": p.name, "name": meta.get("name", p.name), "word": meta.get("word", ""),
                    "count": len(list((p / "images").glob("*.png")))})
    return out


@app.post("/api/train/projects")
def new_project(req: NewProject):
    name = " ".join(req.name.split())[:60]
    word = re.sub(r"[^a-zA-Z0-9]", "", req.word or "").lower()[:24]
    if not name:
        raise HTTPException(400, "Give the project a name.")
    if len(word) < 3:
        raise HTTPException(400, "The code word needs at least 3 letters or numbers, with no spaces.")
    base = slugify(name)
    slug, n = base, 2
    while (TRAIN_DIR / slug).exists() or (MYMODELS_DIR / f"{slug}.safetensors").exists():
        slug, n = f"{base}-{n}", n + 1
    p = TRAIN_DIR / slug
    for sub in ("images", "sheets", "samples"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    write_json(p / "project.json", {"name": name, "word": word, "captions": {},
                                     "created": datetime.now().isoformat(timespec="seconds")})
    log(f"New training project: {name} ({word})")
    return {"slug": slug}


@app.get("/api/train/projects/{slug}")
def project(slug: str):
    p = project_dir(slug)
    meta = read_json(p / "project.json", {})
    caps = meta.get("captions", {})
    imgs = []
    for f in sorted((p / "images").glob("*.png")):
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        imgs.append({"file": f.name, "caption": caps.get(f.name, ""), "w": w, "h": h})
    return {"slug": slug, "name": meta.get("name"), "word": meta.get("word"), "images": imgs,
            "progress": read_json(p / "progress.json", None)}


class ImageUpload(BaseModel):
    data: str
    kind: str = "image"
    label: str = ""


@app.post("/api/train/projects/{slug}/images")
def add_image(slug: str, req: ImageUpload):
    p = project_dir(slug)
    img = decode_data_url(req.data)
    if req.kind == "sheet":
        name = f"sheet_{datetime.now():%Y%m%d_%H%M%S}.png"
        img.save(p / "sheets" / name)
        return {"file": name}
    if max(img.size) > 1536:
        img.thumbnail((1536, 1536), Image.LANCZOS)
    label = slugify(req.label, 20) if req.label else "img"
    i = 1
    while (p / "images" / f"{label}_{i:02d}.png").exists():
        i += 1
    name = f"{label}_{i:02d}.png"
    img.save(p / "images" / name)
    return {"file": name, "w": img.width, "h": img.height}


class CaptionReq(BaseModel):
    file: str
    caption: str


@app.post("/api/train/projects/{slug}/caption")
def set_caption(slug: str, req: CaptionReq):
    p = project_dir(slug)
    safe_name(req.file)
    with FILE_LOCK:
        meta = read_json(p / "project.json", {})
        meta.setdefault("captions", {})[req.file] = " ".join(req.caption.split())[:200]
        write_json(p / "project.json", meta)
    return {"ok": True}


class FileReq(BaseModel):
    file: str


@app.post("/api/train/projects/{slug}/delete-image")
def delete_train_image(slug: str, req: FileReq):
    p = project_dir(slug)
    f = p / "images" / safe_name(req.file)
    if f.exists():
        remove_safely(f, p / "_Deleted")
    with FILE_LOCK:
        meta = read_json(p / "project.json", {})
        meta.get("captions", {}).pop(req.file, None)
        write_json(p / "project.json", meta)
    return {"ok": True}


@app.post("/api/train/projects/{slug}/delete")
def delete_project(slug: str):
    p = project_dir(slug)
    if TRAIN["slug"] == slug:
        raise HTTPException(409, "That project is training right now. Stop it first.")
    remove_safely(p, TRAIN_DIR / "_Deleted")
    return {"ok": True}


@app.get("/train-files/{slug}/{sub}/{name}")
def train_file(slug: str, sub: str, name: str):
    if sub not in ("images", "samples", "sheets"):
        raise HTTPException(404, "Not found")
    f = project_dir(slug) / sub / safe_name(name)
    if not f.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(f, media_type="image/png")


# ================================================================ training runs
class StartReq(BaseModel):
    length: str = "standard"


@app.post("/api/train/projects/{slug}/start")
def start_training(slug: str, req: StartReq):
    p = project_dir(slug)
    if TRAIN["proc"] is not None and TRAIN["proc"].poll() is None:
        raise HTTPException(409, "Another model is already training.")
    n = len(list((p / "images").glob("*.png")))
    if n < 5:
        raise HTTPException(400, f"Add at least 5 pictures first (you have {n}). 15–40 works best.")
    if STATE["status"] == "loading":
        raise HTTPException(409, "Wait for the engine to finish loading, then try again.")
    steps = LENGTHS.get(req.length, LENGTHS["standard"])
    meta = read_json(p / "project.json", {})
    for f in ("control.txt", "progress.json"):
        (p / f).unlink(missing_ok=True)
    for old in (p / "samples").glob("*.png"):
        old.unlink(missing_ok=True)

    unload_engine()
    if STATE["status"] == "busy":
        raise HTTPException(409, "Wait for the current job to finish, then try again.")
    STATE.update(status="training", message=f"Training “{meta.get('name', slug)}”. Making pictures is paused until it finishes.", mode="")
    cmd = [sys.executable, str(APP_DIR / "trainer.py"), "--project", str(p), "--steps", str(steps),
           "--out", str(MYMODELS_DIR / f"{slug}.safetensors")]
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)        # the trainer may need to fetch the one-time training add-on
    with open(p / "train_log.txt", "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S}  {steps} steps, {n} images =====\n")
    lf = open(p / "train_log.txt", "a", encoding="utf-8")
    TRAIN["proc"] = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                     creationflags=NO_WINDOW)
    TRAIN["slug"] = slug
    log(f"Training started: {slug}, {steps} steps, {n} images")

    def watch(proc, logfile):
        proc.wait()
        logfile.close()
        prog = read_json(p / "progress.json", {}) or {}
        if prog.get("status") not in ("done", "cancelled", "error"):
            prog.update(status="error", message="Training stopped unexpectedly. Send train_log.txt (in the project folder) to Claude.")
            write_json(p / "progress.json", prog)
        log(f"Training finished: {slug} → {prog.get('status')}")
        TRAIN["proc"], TRAIN["slug"] = None, None
        load_engine()

    threading.Thread(target=watch, args=(TRAIN["proc"], lf), daemon=True).start()
    return {"ok": True, "steps": steps}


class StopReq(BaseModel):
    keep: bool = False


@app.post("/api/train/stop")
def stop_training(req: StopReq):
    slug = TRAIN["slug"]
    if not slug:
        raise HTTPException(409, "Nothing is training.")
    (TRAIN_DIR / slug / "control.txt").write_text("keep" if req.keep else "stop", encoding="utf-8")
    return {"ok": True}


@app.get("/api/train/status")
def train_status():
    slug = TRAIN["slug"]
    if not slug:
        return {"active": None}
    return {"active": slug, "progress": read_json(TRAIN_DIR / slug / "progress.json", None)}



# ================================================================ scenes (put a character into your background)
def get_inpaint():
    """Inpainting pipeline that shares the already-loaded engine (no extra memory)."""
    if INPAINT["pipe"] is None or INPAINT["src"] is not PIPE:
        from diffusers import FluxInpaintPipeline
        INPAINT["pipe"] = FluxInpaintPipeline.from_pipe(PIPE)
        INPAINT["src"] = PIPE
    return INPAINT["pipe"]


def fit_scene(img: Image.Image, long_side: int = 1152):
    s = long_side / max(img.size)
    w = max(512, int(img.width * s) // 16 * 16)
    h = max(512, int(img.height * s) // 16 * 16)
    return img.resize((w, h), Image.LANCZOS)


class SceneReq(BaseModel):
    bg_data: str = ""
    bg_file: str = ""
    box: dict
    prompt: str
    style: str = "none"
    model: str = ""
    strength: float = 1.0
    outfit: str = ""
    outfit_name: str = ""
    seed: int | None = None


@app.get("/scenes/{name}")
def scene_file(name: str):
    p = SCENES_DIR / safe_name(name)
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")


@app.post("/api/scene")
def scene(req: SceneReq):
    from PIL import ImageFilter
    if STATE["status"] != "ready":
        raise HTTPException(503, STATE["message"])
    prompt = " ".join((req.prompt or "").split())[:1500]
    if not prompt:
        raise HTTPException(400, "Describe who or what to put in the box.")
    if req.bg_file:
        bg_name = safe_name(req.bg_file)
        if not (SCENES_DIR / bg_name).exists():
            raise HTTPException(400, "That background picture is gone. Upload it again.")
        bg = Image.open(SCENES_DIR / bg_name).convert("RGB")
    else:
        bg = decode_data_url(req.bg_data)
        if max(bg.size) > 2048:
            bg.thumbnail((2048, 2048), Image.LANCZOS)
        bg_name = f"bg_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.png"
        bg.save(SCENES_DIR / bg_name)
    try:
        bx, by, bw, bh = (float(req.box[k]) for k in ("x", "y", "w", "h"))
    except Exception:
        raise HTTPException(400, "Draw a box where the character should go.")
    bx, by = max(0.0, min(1.0, bx)), max(0.0, min(1.0, by))
    bw, bh = max(0.0, min(1.0 - bx, bw)), max(0.0, min(1.0 - by, bh))
    if bw < 0.08 or bh < 0.08:
        raise HTTPException(400, "The box is too small. Make it bigger.")

    base = fit_scene(bg)
    W, H = base.size
    pad = 0.03
    x0, y0 = int(max(0, bx - pad) * W), int(max(0, by - pad) * H)
    x1, y1 = int(min(1, bx + bw + pad) * W), int(min(1, by + bh + pad) * H)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
    soft = mask.filter(ImageFilter.GaussianBlur(max(4, int(min(W, H) * 0.015))))

    style = req.style if req.style in STYLES else "none"
    model, model_name, word = "", "", ""
    if req.model:
        if not re.fullmatch(r"[a-z0-9-]+", req.model) or not (MYMODELS_DIR / f"{req.model}.safetensors").exists():
            raise HTTPException(400, "That trained model wasn't found.")
        meta = read_json(MYMODELS_DIR / f"{req.model}.json", {})
        model, model_name, word = req.model, meta.get("name", req.model), meta.get("word", "")
    strength = max(0.0, min(1.5, float(req.strength)))
    run_prompt = f"{word}, {prompt}" if word and word.lower() not in prompt.lower() else prompt
    full = (run_prompt + outfit_clause(req.outfit) +
            ", naturally placed in the scene, matching its lighting, perspective and scale" + STYLES[style])
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)

    with GEN_LOCK:
        if PIPE is None and not FAKE:
            raise HTTPException(503, STATE["message"])
        t0 = time.time()
        try:
            if FAKE:
                time.sleep(0.6)
                gen = base.copy()
                ImageDraw.Draw(gen).rectangle([x0, y0, x1, y1], fill=(200, 120, 220))
                ImageDraw.Draw(gen).text((x0 + 8, y0 + 8), "PRACTICE\n" + prompt[:40], fill=(255, 255, 255))
            else:
                import torch
                apply_model(model, strength)
                g = torch.Generator("cpu").manual_seed(seed)
                gen = get_inpaint()(prompt=full, image=base, mask_image=mask, width=W, height=H, strength=1.0,
                                    num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256,
                                    generator=g).images[0]
        except Exception as e:
            log("Scene failed:\n" + traceback.format_exc())
            if "out of memory" in str(e).lower():
                raise HTTPException(500, "The graphics card ran out of memory. Close games or other heavy apps and try again.")
            raise HTTPException(500, "Something went wrong making that scene. Send studio_log.txt to Claude.")
        secs = round(time.time() - t0, 1)

    out = Image.composite(gen.convert("RGB").resize((W, H)), base, soft)   # background stays exactly as uploaded
    name = f"{datetime.now():%Y-%m-%d_%H%M%S}_scene-{slugify(prompt, 30)}_{seed}.png"
    meta = PngImagePlugin.PngInfo()
    for k, v in (("prompt", run_prompt), ("seed", str(seed)), ("engine", "FLUX.1-schnell inpaint"),
                 ("model", model_name or "Standard")):
        meta.add_text(k, v)
    out.save(IMAGES_DIR / name, pnginfo=meta)
    rec = {"file": name, "prompt": prompt, "style": style, "shape": "scene", "w": W, "h": H, "seed": seed,
           "seconds": secs, "created": datetime.now().isoformat(timespec="seconds"),
           "model": model, "model_name": model_name, "strength": strength if model else None,
           "outfit": " ".join((req.outfit or "").split())[:600], "outfit_name": req.outfit_name[:60],
           "scene": {"bg": bg_name, "box": {"x": bx, "y": by, "w": bw, "h": bh}}}
    add_history(rec)
    log(f"Made scene {name} in {secs}s" + (f" with {model_name}" if model else ""))
    return rec


# ================================================================ outfits (copy this outfit)
READ_JOBS: dict = {}


def outfits_list() -> list:
    return read_json(OUTFITS_DIR / "outfits.json", [])


class ReadReq(BaseModel):
    data: str


@app.post("/api/outfits/read")
def outfit_read(req: ReadReq):
    if STATE["status"] == "training":
        raise HTTPException(409, "Copying an outfit is paused while a model is training.")
    if STATE["status"] != "ready":
        raise HTTPException(409, "Wait for Image Studio to be Ready, then try again.")
    img = decode_data_url(req.data)
    if max(img.size) > 1536:
        img.thumbnail((1536, 1536), Image.LANCZOS)
    jid = uuid.uuid4().hex[:10]
    path = OUTFITS_DIR / f"_reading_{jid}.png"
    img.save(path)
    READ_JOBS[jid] = {"status": "running", "message": "Starting the picture reader…"}
    STATE.update(status="busy", message="Reading an outfit. Making pictures is paused for a minute.")
    threading.Thread(target=run_reader, args=(jid, path), daemon=True).start()
    return {"job": jid}


def run_reader(jid: str, path: Path) -> None:
    job = READ_JOBS[jid]
    out = OUTFITS_DIR / f"_result_{jid}.json"
    stat = OUTFITS_DIR / f"_status_{jid}.json"
    reload_needed = False
    try:
        with GEN_LOCK:
            if not FAKE:
                free = 0
                try:
                    import torch
                    free = torch.cuda.mem_get_info()[0]
                except Exception:
                    pass
                if free < 7.5e9 and PIPE is not None:          # not enough room next to the picture engine
                    job["message"] = "Making room on the graphics card…"
                    unload_engine()
                    reload_needed = True
            env = dict(os.environ)
            env.pop("HF_HUB_OFFLINE", None)                      # first use downloads the reader once
            with open(APP_DIR / "reader_log.txt", "a", encoding="utf-8") as lf:
                lf.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
                lf.flush()
                proc = subprocess.Popen([sys.executable, str(APP_DIR / "reader.py"), "--image", str(path),
                                         "--out", str(out), "--status", str(stat)],
                                        cwd=str(APP_DIR), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                        creationflags=NO_WINDOW)
                while proc.poll() is None:
                    st = read_json(stat, None)
                    if st and st.get("message"):
                        job["message"] = st["message"]
                    time.sleep(1)
        res = read_json(out, None) or {"ok": False, "error": "The picture reader stopped unexpectedly. Send reader_log.txt to Claude."}
        if res.get("ok"):
            job.update(status="done", text=res["text"])
            log("Outfit read.")
        else:
            job.update(status="error", message=res.get("error", "Something went wrong."))
    except Exception:
        log("Outfit reader failed:\n" + traceback.format_exc())
        job.update(status="error", message="The picture reader hit an error. Send studio_log.txt to Claude.")
    finally:
        for f in (path, out, stat):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        if reload_needed:
            load_engine()
        elif STATE["status"] == "busy":
            STATE.update(status="ready", message="Ready")


@app.get("/api/outfits/read/{jid}")
def outfit_read_status(jid: str):
    job = READ_JOBS.get(jid)
    if not job:
        raise HTTPException(404, "That job wasn't found.")
    return job


class OutfitSave(BaseModel):
    name: str
    text: str
    data: str = ""


@app.get("/api/outfits")
def outfits():
    return outfits_list()


@app.post("/api/outfits")
def outfit_save(req: OutfitSave):
    name = " ".join(req.name.split())[:60]
    text = " ".join(req.text.split())[:600]
    if not name or not text:
        raise HTTPException(400, "Give the outfit a name and a description.")
    oid = uuid.uuid4().hex[:10]
    thumb = ""
    if req.data:
        try:
            im = decode_data_url(req.data)
            im.thumbnail((320, 320), Image.LANCZOS)
            thumb = f"thumb_{oid}.png"
            im.save(OUTFITS_DIR / thumb)
        except HTTPException:
            thumb = ""
    with FILE_LOCK:
        items = outfits_list()
        items.insert(0, {"id": oid, "name": name, "text": text, "thumb": thumb,
                         "created": datetime.now().isoformat(timespec="seconds")})
        write_json(OUTFITS_DIR / "outfits.json", items)
    return {"id": oid}


@app.post("/api/outfits/{oid}/delete")
def outfit_delete(oid: str):
    with FILE_LOCK:
        items = outfits_list()
        hit = [o for o in items if o.get("id") == oid]
        if not hit:
            raise HTTPException(404, "Outfit not found")
        write_json(OUTFITS_DIR / "outfits.json", [o for o in items if o.get("id") != oid])
    if hit[0].get("thumb") and (OUTFITS_DIR / hit[0]["thumb"]).exists():
        remove_safely(OUTFITS_DIR / hit[0]["thumb"], OUTFITS_DIR / "_Deleted")
    return {"ok": True}


@app.get("/outfit-thumbs/{name}")
def outfit_thumb(name: str):
    p = OUTFITS_DIR / safe_name(name)
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")


# ================================================================ characters (Character Creator cards)
class CharSave(BaseModel):
    id: str = ""
    name: str
    fields: dict
    prompt: str


@app.get("/api/characters")
def characters():
    return read_json(CHARS_FILE, [])


@app.post("/api/characters")
def character_save(req: CharSave):
    name = " ".join(req.name.split())[:60]
    if not name:
        raise HTTPException(400, "Give the character a name.")
    fields = {str(k)[:40]: str(v)[:200] for k, v in (req.fields or {}).items()}
    with FILE_LOCK:
        items = read_json(CHARS_FILE, [])
        cid = req.id if any(c.get("id") == req.id for c in items) else uuid.uuid4().hex[:10]
        card = {"id": cid, "name": name, "fields": fields, "prompt": " ".join(req.prompt.split())[:1500],
                "updated": datetime.now().isoformat(timespec="seconds")}
        items = [card] + [c for c in items if c.get("id") != cid]
        write_json(CHARS_FILE, items)
    return {"id": cid}


@app.post("/api/characters/{cid}/delete")
def character_delete(cid: str):
    with FILE_LOCK:
        items = read_json(CHARS_FILE, [])
        if not any(c.get("id") == cid for c in items):
            raise HTTPException(404, "Character not found")
        write_json(CHARS_FILE, [c for c in items if c.get("id") != cid])
    return {"ok": True}


@app.post("/api/train/projects/{slug}/from-gallery")
def train_from_gallery(slug: str, req: FilesReq):
    p = project_dir(slug)
    added = 0
    for n in req.files:
        src = IMAGES_DIR / safe_name(n)
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        if max(img.size) > 1536:
            img.thumbnail((1536, 1536), Image.LANCZOS)
        i = 1
        while (p / "images" / f"gallery_{i:02d}.png").exists():
            i += 1
        img.save(p / "images" / f"gallery_{i:02d}.png")
        added += 1
    return {"added": added}


# ================================================================ launch
def mark_interrupted_runs():
    for p in TRAIN_DIR.iterdir():
        prog = read_json(p / "progress.json", None) if p.is_dir() else None
        if prog and prog.get("status") not in ("done", "cancelled", "error"):
            prog.update(status="error", message="Training was interrupted because Image Studio was closed.")
            write_json(p / "progress.json", prog)


def already_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def tray_icon_image():
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, 60, 60], radius=14, fill=(155, 123, 255))
    d.ellipse([18, 18, 46, 46], fill=(242, 184, 75))
    return img


def quit_everything():
    proc = TRAIN.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            (TRAIN_DIR / TRAIN["slug"] / "control.txt").write_text("stop", encoding="utf-8")
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    log("Image Studio closed from the tray.")
    os._exit(0)


def run_tray(url: str) -> None:
    """Small icon by the clock instead of a black window."""
    try:
        import pystray
    except Exception:
        log("Tray icon unavailable (pystray missing); running without it.")
        threading.Event().wait()
        return
    open_folder_fn = lambda which: (lambda *_: os.startfile(str({"images": IMAGES_DIR, "app": APP_DIR}[which])))
    quit_label = lambda _item: "Quit (stops training!)" if TRAIN["slug"] else "Quit Image Studio"
    menu = pystray.Menu(
        pystray.MenuItem("Open Image Studio", lambda *_: webbrowser.open(url), default=True),
        pystray.MenuItem("Open My Images folder", open_folder_fn("images")),
        pystray.MenuItem("Open Image Studio folder", open_folder_fn("app")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(quit_label, lambda icon, _item: (icon.stop(), quit_everything())),
    )
    try:
        pystray.Icon("ImageStudio", tray_icon_image(), f"Image Studio v{VERSION}", menu).run()
    except Exception:
        log("Tray icon failed; Image Studio keeps running. Use 'Start Image Studio (show window)' next time.\n" + traceback.format_exc())
        threading.Event().wait()


def main():
    tray = "--tray" in sys.argv
    if tray and (sys.stdout is None or sys.stderr is None):      # pythonw has no console to print to
        console = open(APP_DIR / "console_log.txt", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or console
        sys.stderr = sys.stderr or console
    url = f"http://{HOST}:{PORT}"
    if already_running():
        print("Image Studio is already running — opening it in your browser.")
        webbrowser.open(url)
        return
    log(f"Image Studio v{VERSION} starting at {url}" + (" (practice mode)" if FAKE else "") + (" in the tray" if tray else ""))
    mark_interrupted_runs()
    threading.Thread(target=load_engine, daemon=True).start()
    if os.environ.get("IMAGE_STUDIO_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    import uvicorn
    if tray:
        server = threading.Thread(target=lambda: uvicorn.run(app, host=HOST, port=PORT, log_level="warning",
                                                             log_config=None), daemon=True)
        server.start()
        run_tray(url)
        quit_everything()
    else:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

"""Image Studio installer step: check the graphics card and download the FLUX.1 Schnell engine."""
import os
import sys
import shutil
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
TARGET = MODELS_DIR / "flux-schnell"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))   # sign-in token is kept here, inside the app folder
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO = "black-forest-labs/FLUX.1-schnell"
PAGE = "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
# Only the parts the app uses (skips the duplicate single-file copy, saves ~24 GB).
PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*", "LICENSE*", "README.md"]
NEEDED_GB = 45


def is_gated(msg: str) -> bool:
    return any(x in msg for x in ("401", "403", "gated", "Gated", "authorized", "awaiting a review"))


def ask_for_token(first: bool) -> bool:
    print()
    print("    ------------------------------------------------------------")
    if first:
        print("    The engine's download page needs a free Hugging Face sign-in.")
        print("    One-time steps (about 2 minutes):")
        print()
        print("    1. Go to  huggingface.co/join  and make a free account.")
        print(f"    2. Go to  {PAGE}")
        print("       and click the button to agree and get access.")
        print("    3. Go to  huggingface.co/settings/tokens  > Create new token")
        print("       > choose \"Read\" > name it  image-studio  > Create > Copy.")
        print()
        print("    Paste the token HERE in this window, not in a chat.")
    else:
        print("    Still blocked. Check that you clicked the agree / access button on")
        print(f"    {PAGE}")
        print("    while signed in, then paste your token again.")
    print("    (Right-click pastes in this window. Type Q and Enter to stop.)")
    print("    ------------------------------------------------------------")
    import getpass
    print("    (The token stays hidden while you paste it - that is normal.)")
    tok = getpass.getpass("    Token: ").strip()
    if not tok or tok.lower() == "q":
        return False
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        print("    Signed in.")
        return True
    except Exception as e:
        print(f"    That token didn't work ({str(e)[:150]}). Copy it again and retry.")
        return ask_for_token(False)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"    Graphics card found: {torch.cuda.get_device_name(0)}")
        else:
            print("    NOTE: The graphics card isn't visible yet. If Image Studio later says so,")
            print("    update your NVIDIA driver and restart the PC. Continuing the download...")
    except Exception as e:
        print(f"    NOTE: couldn't check the graphics card ({e}). Continuing...")

    free_gb = shutil.disk_usage(MODELS_DIR).free / 1e9
    if free_gb < NEEDED_GB and not (TARGET / "transformer").exists():
        print(f"\n    Not enough free disk space: {free_gb:.0f} GB free, about {NEEDED_GB} GB needed.")
        return 1

    from huggingface_hub import snapshot_download
    token_tries = 0
    net_tries = 0
    while True:
        try:
            snapshot_download(REPO, local_dir=str(TARGET), allow_patterns=PATTERNS)
            break
        except Exception as e:
            msg = str(e)
            if is_gated(msg):
                if token_tries >= 3:
                    print("\n    Still can't get access. Send Claude a screenshot of this window.")
                    return 2
                if not ask_for_token(first=(token_tries == 0)):
                    return 2
                token_tries += 1
                continue
            net_tries += 1
            print(f"\n    Download interrupted (try {net_tries} of 5): {msg[:200]}")
            if net_tries >= 5:
                print("    Check your internet connection and run Install again. It picks up where it left off.")
                return 1
            time.sleep(10 * net_tries)

    if not (TARGET / "model_index.json").exists():
        print("    The download finished but files are missing. Run Install again.")
        return 1
    print("    Picture engine downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

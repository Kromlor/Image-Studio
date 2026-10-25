"""Installs extra parts that must exactly match the installed PyTorch (currently: torchvision for the outfit reader).
Safe to run every start: does nothing if everything is already in place, and never changes PyTorch itself."""
import importlib.util
import re
import subprocess
import sys


def works() -> bool:
    r = subprocess.run([sys.executable, "-c", "import torch, torchvision; torchvision.ops.nms; print('ok')"],
                       capture_output=True, text=True)
    return r.returncode == 0 and "ok" in r.stdout


def main() -> int:
    if importlib.util.find_spec("torchvision") is not None:
        if works():
            return 0
        print("  Removing a torchvision that doesn't match...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "--quiet"])
    import torch
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\+(\w+))?", torch.__version__)
    if not m:
        return 0
    major, minor, patch, cuda = int(m[1]), int(m[2]), int(m[3]), m[4] or "cu128"
    if major != 2:
        return 0
    tv = f"0.{minor + 15}.{patch}"          # torch 2.N.x pairs with torchvision 0.(N+15).x
    print(f"  Adding the picture reader's helper (torchvision {tv})...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", f"torchvision=={tv}", "--no-deps", "--quiet",
                        "--index-url", f"https://download.pytorch.org/whl/{cuda}"])
    if r.returncode != 0 or not works():
        print("  Couldn't add it right now (Copy an outfit won't work yet). Everything else is fine.")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "--quiet"],
                       capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

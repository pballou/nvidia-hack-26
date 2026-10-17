"""Pre-resize screenshots once with common/grounding.py:preprocess_screenshot.

Training (training/sft.py) and the Gym rows (data_prep/gym_rows.py) both read the
cached PNGs, so the model sees identical pixels in training and eval, and vLLM's
processor (capped at the same MAX_PIXELS) never resizes them again.
"""

from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from common import grounding

DEFAULT_CACHE = Path("/home/ubuntu/data/cache/preprocessed")


def _resize_one(job: tuple[Path, Path, int]) -> None:
    src, dst, max_pixels = job
    if dst.exists():
        return
    grounding.MAX_PIXELS = max_pixels  # set per worker; preprocess reads it at call time
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")
    with Image.open(src) as im:
        grounding.preprocess_screenshot(im).save(tmp)
    tmp.rename(dst)  # never leave a half-written PNG under the final name


def preresize(image_paths: set[str], root: Path, cache: Path = DEFAULT_CACHE,
              max_pixels: int = grounding.MAX_PIXELS) -> dict[str, str]:
    """Resize each screenshot (relative to root) once; returns image_path -> cached PNG path."""
    out = {p: cache / str(max_pixels) / root.name / Path(p).with_suffix(".png") for p in image_paths}
    with Pool(16) as pool:
        pool.map(_resize_one, [(root / p, dst, max_pixels) for p, dst in out.items()], chunksize=8)
    return {p: str(dst) for p, dst in out.items()}

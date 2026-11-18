"""Fetch the pinned public training file and verify its contents."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from shutil import copyfileobj

URL = ("https://raw.githubusercontent.com/aaronteork/kaggle-binary_classification_bank_churn/"
       "911becd02278325cfaeaaeb8bb55e79bb9635a66/data/train.csv")
SHA256 = "ff1e0eb402d364adaf0c3f9cc8b2351cde0657642bf4cb3ed56efcf31b67d1b0"
DATA = Path(__file__).resolve().parent / "data/train.csv"


def verify(path):
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != SHA256:
            raise ValueError(f"Unexpected contents in {path}. Check the dataset source.")


if __name__ == "__main__":
    if DATA.exists():
        verify(DATA)
    else:
        DATA.parent.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=DATA.parent, delete=False) as tmp:
            staged = Path(tmp.name)
        try:
            with urlopen(URL, timeout=90) as response, staged.open("wb") as output:
                copyfileobj(response, output)
            verify(staged)
            staged.replace(DATA)
        finally:
            staged.unlink(missing_ok=True)
    print(f"Verified {DATA}")

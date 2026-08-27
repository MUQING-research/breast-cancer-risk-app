"""
upload.py — Push hf_breast_cancer to HuggingFace Space
Usage: python upload.py [message]
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "muqing-research/breast-cancer-classification"
FILES   = [
    "breast_cancer_app.py", "theme.css", "app.py", "requirements.txt",
    "Dockerfile",
]

msg = sys.argv[1] if len(sys.argv) > 1 else "Update"
api = HfApi()
folder = Path(__file__).parent

for f in FILES:
    api.upload_file(
        path_or_fileobj=str(folder / f),
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message=msg,
    )
    print(f"  uploaded: {f}")

print(f"\nDone — https://huggingface.co/spaces/{REPO_ID}")

from huggingface_hub import snapshot_download
from pathlib import Path

# Place you Hugging Face Access Token here
hf_token = "your hf access token"

# Define the model repository and storage path
REPO_ID = "mistralai/Mamba-Codestral-7B-v0.1"
DIR = Path.home().joinpath("mistral_models", "Mamba-Codestral-7B-v0.1")

# Ensure the directory exists
DIR.mkdir(parents=True, exist_ok=True)

# Download specific files from the model repository
print(f"Downloading model files from {REPO_ID}...")
snapshot_download(
    repo_id=REPO_ID,
    allow_patterns=["*"],
    local_dir=DIR,
    token=hf_token  # Authenticate with Hugging Face
)

print(f"Model downloaded successfully")

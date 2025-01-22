from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get variables from .env
hf_token = os.getenv('HF_TOKEN')

# Define the model repository and local storage path
REPO_ID = "mistralai/Mamba-Codestral-7B-v0.1"
LOCAL_DIR = Path.home().joinpath("mistral_models", "Mamba-Codestral-7B-v0.1")


# Ensure the local directory exists
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Download specific files from the model repository
print(f"Downloading model files from {REPO_ID}...")
snapshot_download(
    repo_id=REPO_ID,
    allow_patterns=[
        "params.json",  # Model configuration
        "consolidated.safetensors",  # Model weights
        "tokenizer.model.v3"  # Tokenizer file
    ],
    local_dir=LOCAL_DIR,
    token=hf_token  # Authenticate with Hugging Face
)

print(f"Model downloaded successfully and stored at {LOCAL_DIR}")

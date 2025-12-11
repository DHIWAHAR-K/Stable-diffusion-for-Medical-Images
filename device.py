import torch

def get_device():
    """
    Automatically detects the best available accelerator.
    Returns: "mps" (Apple Silicon), "cuda" (NVIDIA), or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    print(f"Detected device: {get_device()}")

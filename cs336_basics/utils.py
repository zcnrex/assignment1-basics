import torch
import numpy.typing as npt
import numpy as np

DEBUG = 6


def print_(msg: str, values: object, level=0):
    if DEBUG == level:
        print(f"{msg} {values}")


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_index = len(dataset) - context_length
    if max_start_index <= 0:
        raise ValueError("Dataset too small for the requested context length.")

    start_indices = np.random.randint(0, max_start_index, size=batch_size)

    x_batch = np.stack([dataset[i : i + context_length] for i in start_indices])
    y_batch = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    x_tensor = torch.tensor(x_batch, device=device)
    y_tensor = torch.tensor(y_batch, device=device)

    return x_tensor, y_tensor


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save model and optimizer state.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        iteration (int): Current iteration or epoch number.
        out (str): File path to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
    print(f"Checkpoint saved to {out}")


def load_checkpoint(src, model, optimizer=None):
    """
    Load model and (optionally) optimizer state.

    Args:
        src (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.

    Returns:
        int: The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {src}")
    return checkpoint.get("iteration", 0)

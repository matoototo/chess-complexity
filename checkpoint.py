import torch


def save(steps, model_state, optim_state, used_files, model_args, path):
    torch.save({
        'steps': steps,
        'model_state': model_state,
        'optim_state': optim_state,
        'model_args': model_args,
        'used_files': used_files
    }, path)


def load(path):
    return torch.load(path)

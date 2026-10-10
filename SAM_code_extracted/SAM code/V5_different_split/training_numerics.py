"""Fail before an invalid update; keep a small diagnostic without saving bad weights."""
import json
from pathlib import Path
import torch


def numerical_failure(output_dir, message, **context):
    report = dict(message=message, **context)
    path = Path(output_dir) / 'numerical_failure.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    raise FloatingPointError(f'{message}; context={context}; report={path}')


def require_finite(value, name, output_dir, **context):
    if not torch.isfinite(value).all():
        numerical_failure(output_dir, f'Non-finite {name}',
                          tensor_shape=list(value.shape),
                          nonfinite_count=int((~torch.isfinite(value)).sum()), **context)


def clip_finite_gradients(model, max_norm, output_dir, **context):
    try:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        # The optimizer and teacher have NOT been updated at this point.
        bad = [name for name, parameter in model.named_parameters()
               if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        numerical_failure(output_dir, 'Non-finite gradient norm before optimizer.step',
                          parameters=bad[:20], detail=str(error), **context)

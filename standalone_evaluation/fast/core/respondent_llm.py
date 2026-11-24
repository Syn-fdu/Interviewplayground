"""Isolated validation dependency, same MODEL_NAME_RE rule as Playground v1.7.5."""
import re
MODEL_NAME_RE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}$")
class LocalModelError(ValueError):
    pass
def validate_model(name, default):
    model=name if name is not None else default
    if not isinstance(model,str) or not MODEL_NAME_RE.fullmatch(model):
        raise LocalModelError("模型名称不合法，请使用 ollama list 中的名称。")
    return model

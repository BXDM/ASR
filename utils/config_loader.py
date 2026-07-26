import yaml
import os


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # secrets.yaml 不进 git（见 .gitignore），单独存放 DeepSeek 等 API key
    secrets_path = os.path.join(os.path.dirname(path), "secrets.yaml")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}
        api_key = (secrets.get("deepseek") or {}).get("api_key")
        if api_key:
            config.setdefault("correction", {})["api_key"] = api_key

    return config

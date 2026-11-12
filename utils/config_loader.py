import yaml
import os


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 单人项目：key 默认直接写在 config.yaml 里（跟仓库走，换机器即用）。
    # secrets.yaml 可选：存在时只「填补空缺」——config.yaml 里已填的 key 优先，
    # 不会被覆盖；没填的才从 secrets.yaml 取。想分离 key 的人才需要它，不进 git。
    secrets_path = os.path.join(os.path.dirname(path), "secrets.yaml")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}

        off = config.setdefault("refine", {}).setdefault("offline", {})
        volc_file_key = (secrets.get("volc_file") or {}).get("api_key")
        if volc_file_key and not off.get("api_key"):
            off["api_key"] = volc_file_key
        dashscope_key = (secrets.get("dashscope") or {}).get("api_key")
        if dashscope_key and not off.get("qwen_api_key"):
            off["qwen_api_key"] = dashscope_key

        corr = config.setdefault("correction", {})
        deepseek_key = (secrets.get("deepseek") or {}).get("api_key")
        if deepseek_key and not corr.get("api_key"):
            corr["api_key"] = deepseek_key
        elif dashscope_key and not corr.get("api_key"):
            corr["api_key"] = dashscope_key

    return config

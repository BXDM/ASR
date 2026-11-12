import logging
import yaml
import os

logger = logging.getLogger(__name__)

# 加密凭据包里的凭据名 → config 里的 Key 路径（与 security/seal.py 的映射一致）
_SECRET_MAP = {
    "volc_asr_access_key": ("asr", "access_key"),
    "volc_file_api_key":   ("refine", "offline", "api_key"),
    "dashscope_api_key":   ("refine", "offline", "qwen_api_key"),
    "deepseek_api_key":    ("correction", "api_key"),
}


def _set_path(cfg: dict, path: tuple, value) -> None:
    if not value:
        return
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def _apply_sealed_secrets(config: dict) -> None:
    """若存在加密凭据包 security/secrets.enc 且能解锁，用其中的 Key 覆盖 config。

    这样 config.yaml 里可以不放明文 Key（留空即可），换电脑克隆后只要提供解锁口令
    （环境变量 ASR_SECRETS_PASSPHRASE 或 security/.passphrase）就能恢复全部凭据。
    解锁失败（无文件/缺口令/口令错/未装 cryptography）都不抛出，沿用 config.yaml 原值。
    """
    try:
        from security.secret_provider import available, get_all
    except Exception:
        return
    try:
        if not available():
            return
        secrets = get_all()
    except Exception as e:
        logger.warning("加密凭据包未能解锁，沿用 config.yaml 内的 Key: %s", e)
        return
    for name, path in _SECRET_MAP.items():
        _set_path(config, path, secrets.get(name))


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

    # 加密凭据包优先级最高：存在且能解锁时覆盖以上来源（config.yaml 可不放明文 Key）。
    _apply_sealed_secrets(config)

    return config

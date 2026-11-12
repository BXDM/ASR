"""
把明文 API Key 加密成 security/secrets.enc（可提交到仓库）。

在项目根目录运行：

  # 方式 A：从当前 config.yaml 里现有的 Key 一键迁移（最省事）
  python -m security.seal --from-config

  # 方式 B：自己准备明文 JSON
  #   1) 复制 security/secrets.plain.example.json 为 security/secrets.plain.json 并填 Key
  #   2) python -m security.seal
  #   （secrets.plain.json 已在 .gitignore，不会进仓库；用完可删）

  # 验证 secrets.enc（只打印 Key 名、不打印值）
  python -m security.seal --show

口令来源：环境变量 ASR_SECRETS_PASSPHRASE，否则交互式输入。
生成后：设 ASR_SECRETS_PASSPHRASE，或把口令写进 security/.passphrase（不进仓库），
程序启动即可自动解锁；随后可把 config.yaml 里的明文 Key 清空。
"""

import getpass
import json
import os
import sys
from pathlib import Path

from security.secret_provider import (
    seal_secrets, open_sealed, _ENC_PATH, _SECURITY_DIR, _PASSPHRASE_ENV,
)

_PROJECT_ROOT = _SECURITY_DIR.parent
_PLAIN = _SECURITY_DIR / "secrets.plain.json"
_CONFIG = _PROJECT_ROOT / "config.yaml"

# config.yaml 里的 Key 路径 → 凭据名（与 utils/config_loader 的覆盖映射一致）
_CONFIG_MAP = {
    "volc_asr_access_key": ("asr", "access_key"),
    "volc_file_api_key":   ("refine", "offline", "api_key"),
    "dashscope_api_key":   ("refine", "offline", "qwen_api_key"),
    "deepseek_api_key":    ("correction", "api_key"),
}


def _passphrase(confirm: bool = False) -> str:
    env = os.environ.get(_PASSPHRASE_ENV)
    if env:
        return env
    pw = getpass.getpass("设置/输入解锁口令: ")
    if not pw:
        print("口令不能为空"); sys.exit(1)
    if confirm and pw != getpass.getpass("再输一次确认: "):
        print("两次输入不一致"); sys.exit(1)
    return pw


def _dig(cfg: dict, path: tuple):
    d = cfg
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _secrets_from_config() -> dict:
    import yaml
    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    out = {}
    for name, path in _CONFIG_MAP.items():
        val = _dig(cfg, path)
        if val:
            out[name] = val
    return out


def main():
    args = sys.argv[1:]

    if "--show" in args:
        secrets = open_sealed(_ENC_PATH, _passphrase())
        print(f"secrets.enc 含 {len(secrets)} 个凭据:", ", ".join(sorted(secrets)))
        return

    if "--from-config" in args:
        secrets = _secrets_from_config()
        if not secrets:
            print("config.yaml 里没找到可迁移的 Key"); sys.exit(1)
        print("将从 config.yaml 迁移:", ", ".join(sorted(secrets)))
    else:
        if not _PLAIN.exists():
            print(f"请先创建 {_PLAIN}（参考 secrets.plain.example.json），或用 --from-config")
            sys.exit(1)
        secrets = json.loads(_PLAIN.read_text(encoding="utf-8"))

    seal_secrets(secrets, _passphrase(confirm=True))
    print(f"已生成 {_ENC_PATH}（{len(secrets)} 个凭据）。该文件可提交到仓库。")
    print("下一步：设 ASR_SECRETS_PASSPHRASE 或写 security/.passphrase；随后清空 config.yaml 里的明文 Key。")
    print("切勿提交：明文口令、security/.passphrase、security/secrets.plain.json。")


if __name__ == "__main__":
    main()

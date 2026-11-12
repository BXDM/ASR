"""
统一加密凭据模块（SecretProvider）。

目标：所有 API Key 加密存放在 security/secrets.enc（可随仓库一起提交、跨电脑使用），
用一个「解锁口令」在运行时解密到内存；口令本身绝不进仓库。业务侧只需 get_key(name)。

加密链路（不自造算法）：
    口令 --scrypt(标准库 KDF)--> 32B 密钥 --Fernet(AES128-CBC + HMAC 认证加密)--> 密文

secrets.enc 是一个 JSON 文本文件（git 友好），内含 KDF 盐、参数与 Fernet 密文。
解锁口令按以下顺序解析：显式传入 > 环境变量 ASR_SECRETS_PASSPHRASE > security/.passphrase 文件。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from pathlib import Path

_ENC_NAME = "secrets.enc"
_PASSPHRASE_ENV = "ASR_SECRETS_PASSPHRASE"
_PASSPHRASE_FILE = ".passphrase"          # security/.passphrase（已在 .gitignore，不进仓库）

_SECURITY_DIR = Path(__file__).resolve().parent
_ENC_PATH = _SECURITY_DIR / _ENC_NAME

# scrypt 参数：整段只在启动时解密一次，这点开销可接受；换机迁移时无需改动。
_N, _R, _P, _DKLEN = 1 << 14, 8, 1, 32


def _fernet_key(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    raw = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=_DKLEN)
    return base64.urlsafe_b64encode(raw)


def seal_secrets(secrets: dict, passphrase: str, enc_path: Path = _ENC_PATH) -> None:
    """把明文 {name: key} 字典加密写入 secrets.enc。"""
    from cryptography.fernet import Fernet
    salt = os.urandom(16)
    f = Fernet(_fernet_key(passphrase, salt, _N, _R, _P))
    token = f.encrypt(json.dumps(secrets, ensure_ascii=False).encode("utf-8"))
    blob = {
        "version": 1,
        "kdf": "scrypt",
        "n": _N, "r": _R, "p": _P,
        "salt": base64.b64encode(salt).decode("ascii"),
        "ciphertext": token.decode("ascii"),
    }
    enc_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")


def open_sealed(enc_path: Path, passphrase: str) -> dict:
    """用口令解密 secrets.enc，返回 {name: key} 字典。口令错/文件损坏抛 ValueError。"""
    from cryptography.fernet import Fernet, InvalidToken
    blob = json.loads(enc_path.read_text(encoding="utf-8"))
    salt = base64.b64decode(blob["salt"])
    n, r, p = blob.get("n", _N), blob.get("r", _R), blob.get("p", _P)
    f = Fernet(_fernet_key(passphrase, salt, n, r, p))
    try:
        data = f.decrypt(blob["ciphertext"].encode("ascii"))
    except InvalidToken as e:
        raise ValueError("解锁口令错误，或 secrets.enc 已损坏") from e
    return json.loads(data.decode("utf-8"))


def _resolve_passphrase(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get(_PASSPHRASE_ENV)
    if env:
        return env
    pf = _SECURITY_DIR / _PASSPHRASE_FILE
    if pf.exists():
        return pf.read_text(encoding="utf-8").strip() or None
    return None


class SecretProvider:
    """按需解锁、内存缓存的凭据读取器。线程安全。"""

    def __init__(self, enc_path: Path = _ENC_PATH):
        self._enc_path = enc_path
        self._secrets: dict | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._enc_path.exists()

    def unlock(self, passphrase: str | None = None) -> bool:
        """解密并缓存。无 secrets.enc 返回 False；缺口令抛 RuntimeError；口令错抛 ValueError。"""
        with self._lock:
            if self._secrets is not None:
                return True
            if not self._enc_path.exists():
                return False
            pw = _resolve_passphrase(passphrase)
            if not pw:
                raise RuntimeError(
                    f"需要解锁口令：设置环境变量 {_PASSPHRASE_ENV}，"
                    f"或在 security/{_PASSPHRASE_FILE} 写入口令")
            self._secrets = open_sealed(self._enc_path, pw)
            return True

    def get(self, name: str, default=None):
        if self._secrets is None:
            self.unlock()
        return (self._secrets or {}).get(name, default)

    def all(self) -> dict:
        if self._secrets is None:
            self.unlock()
        return dict(self._secrets or {})


# ── 模块级便捷接口（业务侧直接用这些）─────────────────────────────────────────
_default = SecretProvider()


def get_key(name: str, default=None):
    """读取一个凭据，例如 get_key("deepseek_api_key")。"""
    return _default.get(name, default)


def get_all() -> dict:
    return _default.all()


def unlock(passphrase: str | None = None) -> bool:
    return _default.unlock(passphrase)


def available() -> bool:
    return _default.available

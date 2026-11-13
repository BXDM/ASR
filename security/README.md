# security/ — 加密凭据模块

所有 API Key 加密存放在 `api.key`（**已 gitignore，不进仓库**），运行时用「解锁口令」解密到内存。
业务代码不出现明文 Key；换电脑时把 `security/api.key` 拷过去并提供口令即可恢复全部凭据。

## 组成

| 文件 | 说明 | 进仓库？ |
|---|---|---|
| `secret_provider.py` | 读取/加解密（`get_key(name)`） | ✅ |
| `seal.py` | 把明文 Key 封进 `api.key` 的工具 | ✅ |
| `api.key` | 加密后的全部 Key（scrypt + Fernet） | ❌ gitignore |
| `secrets.plain.example.json` | 明文填写模板 | ✅ |
| `secrets.plain.json` | 你填的明文（临时用） | ❌ gitignore |
| `.passphrase` | 解锁口令 | ❌ gitignore |

## 加解密原理（不自造算法）

```
口令 --scrypt(标准库 KDF)--> 32B 密钥 --Fernet(AES128-CBC + HMAC 认证加密)--> api.key
```

`utils/config_loader.py` 在加载 `config.yaml` 后，若 `api.key` 存在且能解锁，
就把其中的 Key 覆盖进 config（`config.yaml` 里对应字段留空即可）。

## 口令来源（按优先级）

1. 环境变量 `ASR_SECRETS_PASSPHRASE`
2. `security/.passphrase` 文件

解不出（无文件 / 缺口令 / 口令错 / 没装 cryptography）时不报错，自动回退用 `config.yaml` 内的值。

## 常用操作

```bash
# 从现有 config.yaml 迁移并加密（提示设口令）
python -m security.seal --from-config

# 自己准备明文：复制 secrets.plain.example.json 为 secrets.plain.json 填好，再：
python -m security.seal

# 验证（只列 Key 名，不打印值）
python -m security.seal --show
```

换电脑：克隆仓库 → 拷入 `security/api.key` → 设 `ASR_SECRETS_PASSPHRASE` 或写 `security/.passphrase` → 直接运行。

## ⚠️ 安全约定

- `api.key`、`.passphrase`、`secrets.plain.json` 都在 `.gitignore`，**均不进仓库**。
- 密文和口令都不在仓库里，即使仓库公开也不会泄露 Key。
- 用 U 盘 / 网盘等私下方式在电脑间同步 `api.key`；口令单独记住或放密码管理器。
- 提交前可 `git status` 确认没有 `*.key` 文件被加入。

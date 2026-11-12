# security/ — 加密凭据模块

所有 API Key 加密存放在 `secrets.enc`（随仓库提交），运行时用「解锁口令」解密到内存。
业务代码不出现明文 Key；换电脑克隆后，提供口令即可恢复全部凭据。

## 组成

| 文件 | 说明 | 进仓库？ |
|---|---|---|
| `secret_provider.py` | 读取/加解密（`get_key(name)`） | ✅ |
| `seal.py` | 把明文 Key 封进 `secrets.enc` 的工具 | ✅ |
| `secrets.enc` | 加密后的全部 Key（scrypt + Fernet） | ✅ |
| `secrets.plain.example.json` | 明文填写模板 | ✅ |
| `secrets.plain.json` | 你填的明文（临时用） | ❌ gitignore |
| `.passphrase` | 解锁口令 | ⚠️ 见下 |

## 加解密原理（不自造算法）

```
口令 --scrypt(标准库 KDF)--> 32B 密钥 --Fernet(AES128-CBC + HMAC 认证加密)--> secrets.enc
```

`utils/config_loader.py` 在加载 `config.yaml` 后，若 `secrets.enc` 存在且能解锁，
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

换电脑：克隆仓库 → 设 `ASR_SECRETS_PASSPHRASE` 或写 `security/.passphrase` → 直接运行。

## ⚠️ 关于 `.passphrase` 随仓库提交的取舍

本项目选择把 `.passphrase` **也提交进仓库**，图的是「克隆下来零配置即可运行」。

**这会使加密失去保密性**：密文（`secrets.enc`）与口令（`.passphrase`）同在一个仓库，
任何能访问该仓库的人都能解出全部 Key——效果等同于直接提交明文 Key。

因此：

- **本仓库必须始终保持私有（private）。**
- 一旦转为公开或被转发，火山 / 千问 / DeepSeek 等全部 Key **立即泄露**，且永久留在 git 历史里，需逐个到各平台控制台重置。
- 想要真正的保密性：把 `.passphrase` 重新加入 `.gitignore`、不提交口令，换电脑时手动提供口令即可。

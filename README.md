# Retrieval

基于 FastAPI 的类目检索服务。检索时组合本地 Sentence Transformers 向量、Chroma 稠密检索和 BM25 关键词检索，再用 Reciprocal Rank Fusion（RRF）融合排序。项目还包含一个独立的同步 worker，用于从 PostgreSQL 全量重建和增量更新索引。

## 运行组成

- **API**：只读加载当前索引，提供检索、健康检查和管理接口。启动时不会自动创建或重建索引。
- **Sync worker**：唯一的索引写入进程。通过 Redis 领取 writer lease、执行全量重建和每日增量同步，并发布新索引版本。
- **Redis**：为 worker 提供 writer lease、任务队列和状态存储；API 通过 Redis Pub/Sub 得知索引切换并重载索引。
- **可选 UI**：`ui.py` 是 Gradio HTTP 客户端，调用已运行的 API，不负责加载模型或索引。

API 只做检索时，Redis 不可用不会阻止读取已有索引，但 Redis 状态/管理接口不可用，API 日志中的索引版本监听器会持续重试。全量重建、增量同步需要 Redis 和 sync worker。

## 环境准备

- 建议使用 Python 3.12，并从项目根目录执行以下命令。
- 安装 `requirements.txt` 中的后端依赖：

```powershell
conda activate Retrieval
python -m pip install -r requirements.txt
```

模型默认读取 `models/paraphrase-multilingual-MiniLM-L12-v2/`。模型权重和缓存由 `.gitignore` 排除，不纳入 Git；本地该目录已有模型时无需下载，否则执行：

```powershell
python -m services.downloader
```

该命令从 Hugging Face 下载配置的模型并保存到本地。可在 `.env` 中设置 `HF_ENDPOINT` 使用镜像。API 启动时只加载本地模型，不会自动下载。

## 配置

项目根目录的 `.env` 可覆盖默认配置。不要将数据库密码提交到版本库或粘贴到公开位置。数据库凭据只在 worker 访问 PostgreSQL 时需要；API 的普通检索不访问数据库。

```dotenv
# 模型与索引
MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
MODEL_DIRECTORY=models
CHROMA_COLLECTION_NAME=category_retrieval
CHROMA_DENSE_WEIGHT=0.4
CHROMA_PERSIST_DIRECTORY=data/chromadb
BM25_INDEX_PATH=data/processed/bm25.pkl
CLEANED_JSONL_PATH=data/processed/cleaned.jsonl
INDEX_VERSIONS_DIRECTORY=data/index-versions
ACTIVE_INDEX_SYMLINK=data/active-index

# PostgreSQL：数据库视图必须存在且账号具有读取权限
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_SCHEMA=public
DB_USER=your_readonly_user
DB_PASSWORD=your_password

# 本地 Redis
REDIS_URL=redis://localhost:6379/0
DAILY_SYNC_TIMEZONE=Asia/Shanghai

# 日志
LOG_DIR=data/logs
LOG_LEVEL=INFO
```

路径相对项目根目录解析。环境变量也可以由当前 Shell 提供；进程环境中的同名变量优先于 `.env`。

数据库读取固定视图 `DB_SCHEMA.v_mro4ai` 和时间字段 `createtime`。视图需提供 `company_code`、`product_name`、`price`、`currency`、`actual_mro_category`、`account_code`、`createtime`；这些字段映射到索引记录中的公司编码、类目名称、价格、币种、类目和类目 ID。增量同步依赖 `createtime` 在新增或修改时更新。

主要配置项：

| 配置项 | 默认值/用途 |
| --- | --- |
| `MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`；模型标识，模型文件从 `MODEL_DIRECTORY` 下以模型末级名称读取 |
| `MODEL_DIRECTORY` | `models` |
| `CHROMA_COLLECTION_NAME` | `category_retrieval`；实际 Chroma collection 名称带有 `_dense` 后缀 |
| `CHROMA_DENSE_WEIGHT` | `0.4`；检索默认向量权重，BM25 权重为 `1 - 向量权重` |
| `CHROMA_PERSIST_DIRECTORY` | `data/chromadb`；旧格式索引目录及无活动 generation 时的回退路径 |
| `BM25_INDEX_PATH` | `data/processed/bm25.pkl`；旧格式 BM25 索引及回退路径 |
| `CLEANED_JSONL_PATH` | `data/processed/cleaned.jsonl`；旧格式清洗数据及回退路径 |
| `INDEX_VERSIONS_DIRECTORY` | `data/index-versions`；不可变索引版本目录 |
| `ACTIVE_INDEX_SYMLINK` | `data/active-index`；指向当前索引版本的符号链接 |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `DAILY_SYNC_TIMEZONE` | `Asia/Shanghai`；每日任务使用的时区 |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_SCHEMA`, `DB_USER`, `DB_PASSWORD` | PostgreSQL 连接配置；默认 schema 为 `public`，应按本机环境配置 |
| `LOG_DIR`, `LOG_LEVEL` | 日志目录和级别；默认 `data/logs`、`INFO` |
| `API_BASE_URL` | 仅供可选 `ui.py` 使用，默认 `http://127.0.0.1:8000` |

## 索引与首次运行

索引包含 Chroma 持久化目录、`bm25.pkl` 和 `cleaned.jsonl`。在当前工作副本中，旧格式索引可能位于 `data/chromadb/` 和 `data/processed/`。`data/` 属于运行时数据并被 `.gitignore` 忽略，新克隆的工作副本不一定包含这些文件。

首次启动 worker 时：

1. 若 `data/active-index` 已存在，worker 使用该版本。
2. 若活动链接不存在但旧格式 Chroma 和 BM25 索引都存在，worker 会复制它们到 `data/index-versions/generation-*`，并尝试创建活动符号链接。
3. 若没有可用索引，worker 会等待手动全量重建。
4. 若已有 BM25 索引没有同步水位，worker 会连接数据库，将视图最大 `createtime` 记录为初始水位；这一步只建立同步基线，不会导入历史数据。需要完整导入时，应触发全量重建。

**Windows 注意事项：** 发布索引依赖原子替换符号链接。Windows 需要启用开发者模式，或为运行 worker 的账户授予创建符号链接的权限。出现 `WinError 1314` 时按 `Ctrl+C` 停止 worker，先解决权限，再重新启动。失败尝试可能在 `data/index-versions/` 留下未激活的 `generation-*` 目录；确认 worker 已停止、活动索引未引用这些目录后，再人工清理。

## 本地启动

### 1. 启动 Redis

Redis 可以本机运行，也可以用 Docker 单独启动 Redis 容器：

```powershell
docker run -d --name retrieval-redis -p 6379:6379 redis:7-alpine
```

如果容器已经创建过，后续使用 `docker start retrieval-redis`。确认 `.env` 中 `REDIS_URL` 与 Redis 地址一致。

### 2. 启动 sync worker

在一个终端中，从项目根目录和已安装依赖的 Python 环境运行：

```powershell
python -m sync_worker
```

worker 应持续运行，并获取 Redis writer lease。首次从旧索引迁移时，确保 Windows 符号链接权限已配置；若需要初始化同步水位，还需要可用的 PostgreSQL 连接。

### 3. 启动 API

在另一个终端中运行：

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

本地开发先用单 worker，便于检查启动日志和索引状态。API 初始化失败时，进程仍可能启动，但 `/health` 会返回 `degraded`，搜索会返回服务不可用；应先处理日志中的模型或索引错误。

### 4. 没有索引时执行全量重建

确保 Redis、worker、API 都已启动，且 `.env` 中数据库连接可用。打开 `http://127.0.0.1:8000/docs`，调用 `POST /admin/rebuild`。接口会将任务放入 Redis 队列，由 worker 从数据库视图构建完整索引并发布新版本。通过 `GET /admin/rebuild/status` 查询状态；完成后 `/health` 应显示索引已初始化且一致。

手动全量重建会将同步水位重置到本次读取到的最大 `createtime`。worker 默认每天 00:00（`DAILY_SYNC_TIMEZONE`）安排增量同步。增量任务按水位和同时间戳边界指纹去重，更新已有聚合计数并加入新记录；不会自动清理源数据删除的记录。需要完整反映删除或历史变更时，使用全量重建。

索引更新通过新建 generation 并切换活动链接发布。旧 generation 不会自动删除，以避免 API 尚在读取旧版本时文件被移除；需自行规划磁盘和清理策略。

## 验证与 API

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

健康状态正常时，响应中的 `status` 为 `ok`，模型可用、Chroma collection 与 BM25 均已加载且索引一致。此检查不代表 PostgreSQL 或 sync worker 可用。

检索示例：

```powershell
$body = @{
  query = "办公用品"
  top_k = 50
  top_n = 5
  vector_weight = 0.4
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -ContentType "application/json" `
  -Body $body
```

`POST /search` 参数：

| 参数 | 默认值 | 约束 |
| --- | --- | --- |
| `query` | 必填 | 非空文本 |
| `top_k` | `50` | 候选数，1 到 200 |
| `top_n` | `5` | 返回数，1 到 20 |
| `vector_weight` | 配置项 `CHROMA_DENSE_WEIGHT`，默认 `0.4` | 0 到 1；0 表示 BM25，1 表示向量检索 |

响应包含归一化查询文本、返回数量和结果。结果字段包括 `category_name`、`company_code`、`price`、`currency`、`category`、`category_id`、`init_text`、`rrf_score`、`count`。文本检索会清理空白并将拉丁字母转为大写；BM25 使用 Jieba 分词。

RRF 对各检索列表中的名次求和，常数为 60：

```text
vector_weight / (60 + dense_rank)
  + (1 - vector_weight) / (60 + bm25_rank)
```

结果按类目名称的最高 RRF 分数排序，再以记录 `count` 作为次级排序依据。

管理接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 检查模型和索引加载/一致性 |
| `POST` | `/admin/rebuild` | 将全量重建任务加入 Redis 队列，返回 `202` |
| `GET` | `/admin/rebuild/status` | 查看全量重建状态 |
| `POST` | `/admin/sync` | 将增量同步任务加入 Redis 队列，返回 `202` |
| `GET` | `/admin/sync/status` | 查看增量同步状态 |

管理接口需要 API 可连接 Redis；重建任务还需要 worker 在线。OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

## 可选 UI

`ui.py` 不包含在后端依赖 `requirements.txt` 中。需要运行 UI 时，另行安装 Gradio 和 HTTPX：

```powershell
python -m pip install gradio httpx
python ui.py
```

先启动 API，再在浏览器打开 `http://127.0.0.1:7860`。如 API 地址不同，可在启动 UI 前设置 `API_BASE_URL` 环境变量。

## 测试

运行项目的 unittest 测试：

```powershell
python -m unittest discover -s tests -v
```

部分增量同步测试会在当前本地清洗数据不存在或记录不足时跳过。

## 目录概览

```text
app.py                  FastAPI 应用
sync_worker.py          唯一的索引写入进程
config.py               环境变量和路径配置
controllers/            HTTP 路由、数据模型和重建服务
services/               数据读取、清洗、索引、检索、同步和 Redis 协调
data/                   本地索引、同步版本和日志；运行时数据，不保证随代码分发
models/                 本地 Sentence Transformers 模型
tests/                  unittest 测试
ui.py                   可选的 Gradio API 客户端
requirements.txt        后端 Python 依赖
```

## Docker 部署

仓库使用一个 `Dockerfile` 构建 API 和 worker 共用的镜像。Compose 通过不同的 `command` 启动不同进程，依赖只安装一次，模型只打包一次。

镜像基于 Python 3.12 slim，使用当前 `requirements.txt` 安装依赖，不安装 Conda。模型目录 `models/` 会打进镜像，不需要挂载；它虽被 Git 忽略，但未被 `.dockerignore` 排除。新克隆仓库或本地没有模型时，必须先运行 `python -m services.downloader`，再执行 `docker compose build`，因为 Dockerfile 会复制本地 `models/` 目录。`.env` 不会复制进镜像或构建上下文；Compose 只将显式引用的环境变量传给 worker，其中 `REDIS_URL` 在容器网络内固定为 `redis://redis:6379`。生产环境请通过部署平台的环境变量或 secrets 提供数据库凭据。

### 数据卷与权限

- 宿主机 `data/` 挂载到两个应用容器的 `/app/data`。worker 以读写方式使用它保存 generation 和 `active-index`；API 以只读方式使用它加载索引。
- API/worker 日志分别写入 `data/logs/api/` 和 `data/logs/worker/`，不写入 API 的只读索引挂载。
- Redis 使用独立的 Docker named volume `redis-data`，并启用 AOF 持久化；不要把 Redis 数据放进 Retrieval 的 `data/`。
- Linux 主机使用非 root UID/GID `10001:10001`。若宿主机目录权限不匹配，可在运行 Compose 前设置 `CONTAINER_UID`、`CONTAINER_GID`，并确保索引、日志目录对该身份有相应读写权限。
- API 启动会确保 `data/processed` 存在。由于 API 的 `data/` 是只读挂载，启动前请在宿主机创建所需目录。
- Compose 默认将镜像内进程设为非 root 用户 `10001:10001`。在 Linux/WSL 中，建议用当前用户的 UID/GID 构建镜像，避免 bind mount 写入权限不匹配；这两个值通过 Compose build args 传给 Dockerfile，不会改变本地 Conda 开发环境。

Windows PowerShell 初始化目录：

```powershell
New-Item -ItemType Directory -Force `
  data/processed, data/index-versions, data/logs/api, data/logs/worker | Out-Null
```

worker 发布版本时需要在 `data/` 下创建临时符号链接并通过原子替换更新 `active-index`。Linux 容器文件系统支持该操作；Windows 宿主机 bind mount 到 NTFS 时，符号链接行为可能受 Docker Desktop/文件系统限制。生产或本地验证优先使用 Linux 主机目录、WSL2 Linux 文件系统或 Docker named volume 保存索引；不要在 Windows NTFS bind mount 上假设符号链接一定可用。若出现 `WinError 1314` 或链接创建失败，worker 无法发布索引。

**API 只读挂载注意：** Compose 将整个 `data/` 以只读方式挂给 API，确保 API 进程不能改写索引。Chroma 使用持久化 SQLite 数据库；若目标 Chroma 版本在打开时需要写入数据库元数据或锁文件，API 可能无法初始化。首次部署应检查 `/health` 和容器日志；不要为了绕过错误直接让多个 API 对同一索引进行写入。

### 构建和启动

推荐从 WSL2 的 Linux 文件系统目录（例如 `/home/<user>/retrieval`）运行，而不是 `/mnt/c` 或 Windows 映射目录，以降低索引符号链接和文件权限问题。Compose 会自动读取项目根目录 `.env` 中的数据库连接变量；`.dockerignore` 排除它只影响镜像构建上下文，不影响 Compose 插值。

```bash
mkdir -p data/processed data/index-versions data/logs/api data/logs/worker
docker compose config
docker compose build
docker compose up -d --scale api=1 --scale sync_worker=1
docker compose ps
```

`CONTAINER_UID` 和 `CONTAINER_GID` 通过 Compose build args 传给 Dockerfile，决定容器中的非 root 用户 ID。它们只对当前 shell 会话生效；新终端需重新设置，或写入 Compose 使用的 `.env`。确保该 UID/GID 对宿主机 `data/` 及日志目录有写权限。首次部署没有索引时，不需要预先创建 `chromadb/`、`bm25.pkl` 或 `cleaned.jsonl`；worker 会在全量重建时生成它们。

启动后检查服务、Redis 和 worker：

```bash
docker compose ps
docker compose exec redis redis-cli ping
docker compose logs --tail=100 sync_worker
docker compose exec --index 1 api python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10)))'
```

Redis 应返回 `PONG`。worker 日志应显示获取 writer lease 并进入调度/队列循环；没有索引时，等待手动全量重建是正常状态。已有索引时，API `/health` 应返回 `status: ok` 且索引一致。空索引首次构建期间 API 可能暂时返回 `degraded`；全量重建发布新 generation 后，API 收到 Redis 通知并重载，再检查应恢复正常。worker 若反复出现符号链接或权限错误，则尚未就绪。

没有索引时，需通过可访问 API 的方式提交首次全量重建。当前 Compose 没有发布 API 宿主机端口，可从 API 容器内部提交并查询任务：

```bash
docker compose exec --index 1 api python -c 'import urllib.request; request=urllib.request.Request("http://127.0.0.1:8000/admin/rebuild", method="POST"); print(urllib.request.urlopen(request, timeout=10).read().decode())'
docker compose exec --index 1 api python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/admin/rebuild/status", timeout=10).read().decode())'
```

重建完成后检查 API 健康状态；双 API 部署时分别检查两个副本：

```bash
for index in 1 2; do
  docker compose exec --index "$index" api python -c 'import json, urllib.request; data=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10)); print(data); raise SystemExit(data.get("status") != "ok")'
done
```

Compose 默认启动 Redis、一个 API 和一个 worker。Redis 映射到宿主机 `127.0.0.1:6379`，容器间通过服务名 `redis:6379` 访问。API 在 Compose 网络内监听 `8000`，默认不发布固定宿主机端口，以免妨碍多副本。要在单 API 本地调试时，可在 `api` 服务增加：

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

添加固定端口映射后，API 只能运行一个副本。需要多副本时，去掉该映射，并由外部负载均衡器接入 Compose 网络中的 `api:8000`。API 每个容器仍按要求以 Uvicorn `--workers 1` 运行。

### 扩容和停止

```powershell
# 双 API、双 worker；可按资源和可用性需求调整副本数
docker compose up -d --scale api=2 --scale sync_worker=2

# 停止容器，但保留 data/ 和 Redis named volume
docker compose down
```

`sync_worker=2` 时两个实例通过 Redis lease 互斥，任一时刻至多一个执行索引写任务，另一实例等待并可在 lease 释放后接管。API 副本共享只读索引卷并通过 Redis Pub/Sub 重载版本。API 没有宿主机端口映射，多副本时应由外部负载均衡器接入 Compose 网络的 `api:8000`；若仅启动一个 API，可按上文临时增加端口映射供本机访问。不要使用 `docker compose down -v`，除非确认要删除 Redis 持久化卷。若只需检索已有索引，也可以只运行 API；Redis 和 worker 用于管理接口及索引更新。

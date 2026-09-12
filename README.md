# tsubame-agentsociety2

[AgentSociety 2](https://github.com/tsinghua-fib-lab/AgentSociety) を [TSUBAME 4.0](https://www.t4.cii.isct.ac.jp/) 上で動かす。
LLM バックエンドは同一ノードに立てた [vLLM](https://github.com/vllm-project/vllm)。

## 構成

- vLLM と AgentSociety を同一ジョブ・同一ノードで実行
  - vLLM は `127.0.0.1` で待ち受け → ノード間通信もポート公開も不要
  - 外部依存は OpenAI 互換エンドポイント 1 つのみ
  - 実行結果: `<run_dir>/replay/*.jsonl`

```
scripts/submit.sh jobs/run_sim.sh
  └─ 計算ノード
     ├─ vLLM serve  127.0.0.1:8000   (全 GPU を data parallel で使用)
     └─ python -m agentsociety2.society.cli  →  http://127.0.0.1:8000/v1
```

## 動作環境

実測値。

| 項目         | 値                                              |
| ------------ | ----------------------------------------------- |
| OS           | Red Hat Enterprise Linux 9.4                    |
| スケジューラ | Altair Grid Engine（`qsub` / `qrsh` / `iqrsh`） |
| GPU          | NVIDIA H100 94GB (95,830 MiB)                   |
| ドライバ     | 580.105.08 / CUDA 13.0                          |
| Python       | 3.12.14（uv 管理）                              |
| torch        | 2.11.0+cu130                                    |

| 資源タイプ | GPU     | CPU      | メモリ |
| ---------- | ------- | -------- | ------ |
| `gpu_1`    | H100 x1 | 8 core   | 96GB   |
| `node_q`   | H100 x1 | 48 core  | 192GB  |
| `node_f`   | H100 x4 | 192 core | 768GB  |

- システム Python は 3.9 → AgentSociety 2 の要件（`>=3.11,<3.14`）を満たさない

## 設定

アカウント名・ホスト名・作業パス・予約 ID は**リポジトリに置かない**。

```bash
cp config/env.example config/env.local   # 記入する。git 管理外
```

- `scripts/lib/common.sh` が読み込む
- 以降どのスクリプトも、これらを引数に取らない
- ログを外へ出すときは `scripts/redact.sh` に通す

## セットアップ

1 回実行。番号順・各段独立に再実行可能。

```bash
scripts/sync.sh
scripts/ssh.sh 'bash "$REMOTE_REPO"/scripts/setup/01_install_uv.sh'
scripts/ssh.sh 'bash "$REMOTE_REPO"/scripts/setup/02_sync_env.sh'
scripts/ssh.sh 'bash "$REMOTE_REPO"/scripts/setup/03_download_models.sh'
```

| スクリプト              | 内容                                    |
| ----------------------- | --------------------------------------- |
| `01_install_uv.sh`      | uv を導入（既存なら何もしない）         |
| `02_sync_env.sh`        | `uv.lock` から環境を再現・import を検証 |
| `03_download_models.sh` | モデル重みを事前取得                    |

- 依存は `pyproject.toml` + `uv.lock` で固定 → `uv sync --locked` で再現
- 環境の実体は共有領域に置く（`UV_PROJECT_ENVIRONMENT`）。ホームは容量を共有するため
- モデル重みはジョブ外で取得 → 確保した GPU 時間をダウンロードで消費しない

### 依存構成の制約

vLLM のバージョンを明示指定すると解決できない。

```
agentsociety2 2.8.7  → protobuf <6.0.0
vllm (新しい版)       → nvidia-cutlass-dsl → protobuf >=6.30.2
```

- `vllm` を未固定にする → uv が共存できる最新版を選ぶ
- 現状 0.22.1。Qwen3.6 のモデルカードが要求する `>=0.19.0` を満たす
- 実際の解決結果は `uv.lock` が記録

## 実行

```bash
scripts/submit.sh jobs/smoke.sh      # vLLM だけ検証
scripts/submit.sh jobs/run_sim.sh    # シミュレーション本体
scripts/watch.sh                     # 走っているジョブに合流
```

- `submit.sh`: sync → qsub → ログ追尾を 1 コマンドで実行
  - `AR_ID` があれば予約枠へ、無ければ `-q prior` へ投げ分け
  - `NO_AR=1` で予約を使わない / `NO_WATCH=1` で追尾しない
- `watch.sh`: ジョブ ID 省略時は最新ジョブを自動追従
- `smoke.sh` は vLLM だけを検証し、AgentSociety に進まない
  - ➜ 失敗時に vLLM 側かシミュレータ側かを切り分けられる

資源タイプと実行時間はジョブスクリプト内の `#$` 指示行が既定・`qsub` 引数が優先。

```bash
scripts/submit.sh jobs/run_sim.sh -l node_f=1 -l h_rt=3:00:00 \
  -v MODEL=Qwen/Qwen3.6-35B-A3B-FP8,NUM_AGENTS=16,NUM_ROUNDS=10
```

## パラメータ

`scripts/lib/common.sh` に集約・環境変数で上書き可能。

| 変数                | 既定値              | 意味                                                                        |
| ------------------- | ------------------- | --------------------------------------------------------------------------- |
| `MODEL`             | `Qwen/Qwen3.5-4B`   | vLLM で提供するモデル                                                       |
| `ENV_MODULE`        | `CommonsTragedyEnv` | シナリオの環境モジュール                                                    |
| `NUM_AGENTS`        | `0`                 | `0` は環境ごとの既定に従う                                                  |
| `NUM_ROUNDS`        | `4`                 | 1 ラウンド = run + questionnaire                                            |
| `TICK_SECONDS`      | `900`               | 1 run ステップの模擬秒数                                                    |
| `ENABLE_THINKING`   | `1`                 | エージェントに推論させる。`0` でスループット優先                            |
| `MAX_MODEL_LEN`     | `65536`             | vLLM のコンテキスト長                                                       |
| `MIN_PARTICIPATION` | `0.5`               | 1 ラウンドあたり行動すべきエージェントの割合。下回ると run を失敗扱いにする |

| 用途     | モデル                     | 実測重み | 資源タイプ |
| -------- | -------------------------- | -------- | ---------- |
| 疎通確認 | `Qwen/Qwen3.5-4B`          | 9.3 GB   | `gpu_1`    |
| 本番     | `Qwen/Qwen3.6-35B-A3B-FP8` | 37.5 GB  | `node_f`   |

- 既定を 4B にする理由: パイプラインの不具合を数十秒で表面化させる
- 本番モデルは MoE（総 35B / 活性 3B）
  - 重み 37.5GB に対し、1 トークンあたり動くのは 3B 分だけ
  - ➜ 同じ重みサイズの dense より、同時に捌けるリクエスト数が多い
  - エージェントを増やすほど所要時間は同時処理数で決まる

## 動作確認

`Qwen/Qwen3.5-4B` + `gpu_1` + `CommonsTragedyEnv` 4 エージェント / 4 ラウンドでの実測。

| 指標           | 値                       |
| -------------- | ------------------------ |
| 所要時間       | 14.4 分（vLLM 起動含む） |
| 成立ラウンド   | 4 / 4                    |
| 参加率（平均） | 81%                      |
| プール         | 100 → 20                 |
| ReAct 失敗     | 1 件                     |

```
participation (agents that submitted, out of 4):
  round 1: 3/4  {"Agent-1": 4, "Agent-2": 4, "Agent-3": 8}
  round 2: 4/4  {"Agent-1": 5, "Agent-2": 1, "Agent-3": 10, "Agent-4": 10}
  round 3: 4/4  {"Agent-1": 4, "Agent-2": 3, "Agent-3": 10, "Agent-4": 10}
  round 4: 2/4  {"Agent-2": 1, "Agent-3": 10}
  mean 81%
```

- 環境への採取要求 55 回は**すべて同一文**
  - 値は `variables` で渡るため、テンプレートが 1 種類に収束する
  - ➜ embedding を立てればキャッシュが効く形になっている（現状は 404 で毎回コード生成）
- `jobs/smoke.sh` は exit 0 / `POST /v1/chat/completions` が 200
- 完了判定は exit code だけを見ない
  - agentsociety2 は LLM 呼び出しの失敗を握り潰す経路がある
  - 例: embedding 未設定時は warning に落ちてキャッシュミス扱いで続行
  - ➜ `scripts/check_replay.py` が replay のレコード数まで確認する
- レコード数だけでも足りない
  - ラウンドは 1 体が提出した時点で成立する → 全ラウンド成立 = 全員参加ではない
  - ➜ 同じスクリプトがラウンドごとの参加人数を出し、`MIN_PARTICIPATION` を下回ったら失敗にする

TSUBAME 無しで走る自己チェック。

```bash
uv run --with pytest --no-project --python 3.12 pytest   # scripts/ の生成物と判定
bash scripts/lib/check_common.sh                          # common.sh の導出値
shellcheck -x -P SCRIPTDIR jobs/*.sh scripts/*.sh scripts/lib/*.sh \
  scripts/remote/*.sh scripts/setup/*.sh
```

- テスト対象の 2 本は標準ライブラリしか使わない → `pyproject.toml` に依存を足さずに走る

## TSUBAME で踏んだ落とし穴

いずれも実機で再現・修正済み。

**uv のキャッシュを共有領域に置くとビルドが失敗**

- ソースビルドは隔離環境の Python をキャッシュ内から実行 → 共有領域配下では `Operation not permitted`
- wheel を持たない `stringcase`（agentsociety2 の必須依存）が該当
- ➜ キャッシュはホームに置く

**ログインノードで uv が rayon プールの初期化に失敗**

- `nproc`: 96 / `ulimit -u`: 150
- uv はコア数からスレッド数を決定 → 大きな install の途中で `Resource temporarily unavailable` で abort
- ➜ `UV_CONCURRENT_*` を絞る

**mcp 2.x が agentsociety2 を破壊**

- agentsociety2 2.8.7 の依存指定: `mcp[cli]>=1.13.1`（上限なし）
- mcp 2.x で `FastMCP` → `MCPServer` に改名
- 素直に解決 → `import agentsociety2.society.cli` が `mcp.server.fastmcp` で落ちる
- ➜ `mcp<2` に固定

**`AGENTSOCIETY_LLM_API_KEY` は import 時に検証**

- 検証箇所: CLI 起動時ではなく `agentsociety2.config` の import 時（`config/config.py:492`）
- ➜ モジュールを import するだけでも設定が必要
- vLLM は値を見ない → ローカル実行では placeholder で足りる

**Ray はジョブの割当コア数を見ない**

- `ray.init` の `num_cpus` は既定で `os.cpu_count()` → 割当スロット数に関係なく物理ノードのコア数を返す
- ➜ `AGENTSOCIETY_LLM_RAY_MAX_WORKERS` を `NSLOTS` に合わせる
- `_temp_dir` も未指定 → `/tmp` にフォールバック。`RAY_TMPDIR` をジョブ専用 `TMPDIR` へ向ける

**Grid Engine はジョブスクリプトをコピーして実行**

- 実体は `/var/spool/age/<node>/job_scripts/<jobid>` → `BASH_SOURCE` からの相対パス解決が壊れる
- ➜ `SGE_O_WORKDIR`（投入元ディレクトリ）を使う

**予約枠は `node_f` のみ許可**

- `gpu_1` を指定 → `gpu_1 not permitted in AR` で拒否
- ➜ 小さい疎通確認は `NO_AR=1` で `-q prior` へ

**FlashInfer のカーネルは初回だけ JIT コンパイル**

- Qwen3.5 系のハイブリッド構成は GDN prefill カーネルを JIT 生成・ログを出さずに数分停止
- 実測: 初回 warmup 480.9 秒 → 2 回目 4.8 秒
- キャッシュは `~/.cache/flashinfer/<version>/<arch>/`・ノード間共有でジョブをまたいで残る
- 再コンパイルはバージョン更新・GPU 世代変更・新しいカーネル種別のときだけ
- ➜ 初回のみ `h_rt` を長めに取る。`VLLM_EXTRA_ARGS='--gdn-prefill-backend triton'` で回避も可能

**AgentSociety は tool calling を使う**

- エージェントのステップ呼び出しは毎回 `tool_choice="auto"` 付き
- vLLM 側にフラグが無いと 400 で拒否

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

- litellm がリトライするが、同じ理由で失敗し続ける
- ジョブは exit 0 で終わる。replay が薄くなるだけで、原因はログに埋もれる
  - 実測: フラグ無しで警告 48 件 → フラグ有りで 0 件
- parser は chat template の形で決まる
  - Qwen3.x は `<tool_call><function=...>` を出力 → `qwen3_coder`
  - vLLM 0.22.1 が持つのは `qwen3_coder` と `qwen3_xml` の 2 つ
- ➜ `scripts/check_endpoint.py` が同じ形のリクエストを 1 回投げて確かめる

**thinking 有効時は応答が空で返ることがある**

- Qwen3.5 以降は思考トークンを先に消費
- 予算が尽きる → 思考ブロックが閉じない → `content` も `reasoning_content` も空
- サーバが壊れているように見えるが、単に足りていないだけ
- agentsociety2 は `max_tokens` を指定しない（パッケージ内に出現しない）
  - 応答の長さを縛るのは `--max-model-len` だけ
- ➜ 疎通チェックは thinking 有効時に 1024 トークン確保・`finish_reason` で切り分け
- ➜ `MAX_MODEL_LEN` に余裕を持たせる。ただし KV キャッシュと引き換え

**`ask` と `intervene` はエージェントに届かず、代わりにプランナが勝手に動く**

- `AskStep` / `InterveneStep` は宛先を持たない（`target_agent_ids` があるのは `QuestionnaireStep` だけ）
- CLI は本文を `AgentSocietyHelper` の Plan-and-Execute に渡す
  - プランナは coder ロール → 未設定なら主モデルにフォールバック。つまり同じ 4B
- ヘルパのツールは 4 つ。エージェントに届くのは `ask_agents(agent_ids, question)` だけ
- 本文は何体いるかを伝えない → プランナが id を推測することになる

4 ラウンドの実測では、プランナは `ask_agents` を**一度も選ばなかった**。代わりに `ask_environment` で
自分が採取を提出し、`agent_name` はその場で捏造していた。

```
Task: It is now round 1. First pick a whole number between 1 and 10 ...
📋 Initial Plan (1 steps):
  1. Request 5 units from the shared pool via environment router
  ✓ Result: Successfully submitted an extraction request for 5 units to Agent-1.
```

- replay に残る `Agent-1: 5` は **Agent-1 が出したものではない**
- 実験に 5 人目が混ざっていたことになる。参加者が欠けるより質が悪い
- 同じログに出ていた残りも、すべてプランナ由来

| ログ                                            | 実体                                                                             |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| `Unknown tool: think`                           | プランナが存在しないツールを捏造                                                 |
| `... unexpected keyword argument 'param1'`      | プランニングプロンプトの JSON 例 `"args": {"param1": "value1"}` をそのままコピー |
| `ask_env mutation is disabled in readonly mode` | 届かなかった指示を questionnaire（readonly）で実行しようとした                   |

- 仮に届いても次の `run` には残らない。`agent.ask()` は `memory_runtime.after_step` を呼ばない（呼ぶのは `step()` だけ）
- ➜ `ask` も `intervene` も使わない。`scripts/gen_config.py` は `run` と `questionnaire` しか生成しない
- ➜ ルールも手順もプロファイルに書く。step モードで毎ターン読まれる唯一のテキスト
  - `run` は `AgentSociety.step()` が全 agent id を無条件に fan-out する
  - 環境の observe 系ツールは step ごとに自動で呼ばれる → プール残量は最初から見えている

**`ask_env` に値を文中へ書くのは、仕様に反していて遅い**

`ask_env` のツールスキーマ自身がこう指示している。

> write `instruction` as a reusable template with stable wording, and put
> changing runtime values in `variables` instead of embedding them directly in the text

- コード生成側にも「変わる値は `ctx['variables']` から読め、直値を埋めるな」と指示が入る
- つまり `submit an extraction of {amount} units` は**仕様どおりの書き方**
  - PR #1 で全員が 1 単位になったのは、`variables` に値を渡さなかったから
  - プレースホルダを書いたことが原因ではない
- 数字を文中に書いても動くが、毎回別のテンプレートになる
  - キャッシュが原理的に当たらない → 毎回コード生成の LLM 往復（実測 5〜25 秒）
  - 残さなければならない値を、router が散文から取り出す側に置くことになる
- ➜ instruction は固定文にして変数名を名乗らせ、値は `variables` に入れさせる
  - 上流の `CommonsTragedyAgent` も `agent_name` / `requested_extraction` をこの形で渡している

**禁止したい形を例示すると、モデルはそれを書く**

- role に「`{amount}` や `N` のようなプレースホルダを書くな」と列挙した
- 次の run で `submit an extraction of N units for Agent-1` が出た。禁止文からのコピー
- ➜ 送るべきものを書く。避けるべきものは書かない

**環境はエージェントの言葉から名前を読む。1 文字違うと誰のものでもなくなる**

- 実測: エージェントが `... for Agent 3`（ハイフン無し）で要求
- 環境ツールは渡された `agent_name` のキーを作る → 誰の取り分でもない採取が成立する
- ラウンドは成立し、プールは減り、exit code は 0
- ➜ 名前は「profile のとおり正確に写せ」と指示する
- ➜ `scripts/check_replay.py` が `extractions` のキーを実在のエージェント名と照合する
- ただし**実在する名前を騙られた場合はすり抜ける**
  - 前回 run のヘルパは実在の `Agent-1` を名乗った
  - 事後チェックでは捕まらない。`ask` / `intervene` を置かないことだけが防御になる

**`rsync --delete` は実行中ジョブの出力を消す**

- Grid Engine は投入元ディレクトリに `*.o<jobid>` を書く。`.gitignore` は rsync に効かない
- ➜ `scripts/sync.sh` が除外する。素の rsync を使わない

## 参照

- AgentSociety 2 ドキュメント: <https://agentsociety2.readthedocs.io/>
- vLLM ドキュメント: <https://docs.vllm.ai/>
- TSUBAME 4.0 利用の手引き: <https://www.t4.cii.isct.ac.jp/docs/handbook.ja/>

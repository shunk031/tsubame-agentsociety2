# agentsociety2 のエージェントを動かすための取り決め

どのクラスタで動かしても同じように踏む、agentsociety2 2.8.7 の挙動。
TSUBAME 固有の環境の罠は `README.md` の「TSUBAME で踏んだ落とし穴」にある。

いずれも実機で再現・修正済み。シナリオを設計するとき、あるいは参加率が落ちたときに読む。

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

**どれだけ起きるかはモデル規模で決まる。**

| モデル | ラン数 | 名前が壊れたラン |
| --- | --- | --- |
| `Qwen3.6-35B-A3B-FP8` | 10 | **1** |
| `Qwen3.5-4B` / 16 体 | 9 | 5 |
| `Qwen3.5-4B` / 128 体 | 2 | 2（複数件） |

**35B でもゼロにはならない。** 8 ラン連続で出なかったあと、9 本目（16 体）で `"agent_0002"` が出た。
これはワークスペースのディレクトリ名（`agent_<id:04d>`）で、profile 上の名前ではない。
頻度は大きく下がるが、消えるわけではない。

- 壊れ方は一様ではない
  - `"3"` `"16"` `"5"` `"2055"` `"30"` `"61"` `"70"` `"114"` `"82"` — 数字だけ、あるいは無関係な数
  - `"Agent_60"` `"Agent 3"` `"agent16"` `"Agent-0105"` `"Agent-Agent-64"` — 区切り・大小・桁の崩れ
  - `"agent_0002"` — ワークスペースのディレクトリ名（35B で出たのはこれ）
  - `"cheese"` `"true"` `"statistics_agent"` — 名前と無関係な語。書き損じでは説明できない
- エージェント数ではなくモデルで分かれる。4B は 4 体でも 128 体でも出し、35B は 16 体でも出さない
- ➜ 現状の回避策は大きいモデルを使うこと

**既定のリクエスト上限 60 秒は、待ち行列を作る運用と噛み合わない**

- `config/llm_dispatcher.py` が `AGENTSOCIETY_LLM_REQUEST_TIMEOUT`（既定 `60`）を
  モジュール読み込み時に 1 回だけ読む
- 60 秒という値は「待たされているリクエストは異常」という前提に立っている。
  リモート API 相手ならそれでよい
- **自前のバックエンドを飽和させる運用では前提が逆になる。** 行列ができている状態が
  正常で、vLLM は `Waiting: 13〜42` を報告する
- 1 枚のカードに 100 本以上が同時に載ると、1 本あたりの生成速度は単独時の数十分の一に
  落ちる。ふつうの長さの応答でも、順番待ちを足せば 60 秒を普通に超える
- 実測（128 体 / H100 1 枚 / 27B）

  | | 値 |
  | --- | --- |
  | 5 分間の `litellm.Timeout` | **285 件** |
  | 電力 | 82% → **44%** |

  タイムアウトしたリクエストは再送されるので、混雑の原因に混雑が足される
- ➜ 飽和させるなら `AGENTSOCIETY_LLM_REQUEST_TIMEOUT` を伸ばす。上限そのものは
  本当にハングしたリクエストを拾うために残す。本物の失敗はディスパッチャの
  リトライループの担当なので、大きい値の代償は検知の遅さだけ

**適応セマフォが応答時間のばらつきを輻輳と誤読して、並列度を自分で絞る**

- `config/llm_dispatcher.py` の `AdaptiveSemaphore`（AIMD）が LLM 呼び出しを挟む。プロセスごとに 1 個
- 既定値: 初期 `AGENTSOCIETY_LLM_RAY_CONCURRENCY=16` / `AGENTSOCIETY_LLM_LATENCY_DEGRADE_FACTOR=4.0` /
  `overload_threshold=0.1`（ハードコード）/ `AGENTSOCIETY_LLM_REQUEST_TIMEOUT=60`
- 判定は**相対比**。`latency > baseline × 4.0` を slow と数え、1 ラウンドの 10% が slow なら limit を半減
- エージェントの応答長はステップごとに大きく変わるので、この比は恒常的に超える
  - レート制限は一度も起きない（ログは常に `429=0/N`）
  - 専用のローカルバックエンドに対しては、輻輳ではなく分散を測っているだけ
- **絶対時間を短くしても直らない。** ベースラインも一緒に下がるので、4 倍という比は縮まらない
  - 実測: `ENABLE_THINKING=0` にすると `DECREASE` は **34 回**。thinking 有効の既定（21 回以上）より**増えた**
  - ➜ thinking 固有の問題ではない。応答長が変動するワークロード全般に効く
- 実測（4B / 16 エージェント / 4 ラウンド）

  | | 既定 | `LATENCY_DEGRADE_FACTOR=inf` | thinking off・既定 |
  | --- | --- | --- | --- |
  | `DECREASE` | 21 回以上 | **0 回** | 34 回 |
  | ReAct 失敗 | 48 件 | 12 件 | — |
  | ラウンド所要 | 429 / 224 / 299 / 60 分超 | 368 / 348 / 321 / 322 | — |

- ➜ ローカル vLLM では `AGENTSOCIETY_LLM_LATENCY_DEGRADE_FACTOR=inf` で相対判定を切る
  - コード側に `!= float("inf")` という無効化の分岐がある
- **以下はすべて `max_concurrency=1` での観測**。当時どの env モジュールも
  `is_concurrency_safe()` を宣言しておらず、env router actor が全エージェントの
  環境アクセスを直列化していた。原因と解消は後述の「env actor の直列化」を読む
- 当時の観測: 16 体で `Running` は平均 1〜3 本（容量は 32 並列・`max_num_seqs` 256）
- **128 体でも埋まらなかった。** 1145 サンプル中 `Waiting` がゼロでないのは 2 回だけ

  | 体数 | `Running` 中央値 | 最大 | `Waiting`>0 | KV 最大 |
  | --- | --- | --- | --- | --- |
  | 16 | 1〜3 | — | 0 | 0.7% |
  | 128 | **2** | 48 | 2 / 1113 | 13.8% |
  | 128（embedding 有り） | **3** | 52 | 1 / 664 | 15.4% |
  | 256 | **2** | 80 | 0 / 480 | 19.3% |

  - 頭数を 16 倍にしても飽和しなかった。最大値だけが伸びて定常は薄いまま
  - 256 体の行は wall clock で打ち切られたラン（1 ラウンドのみ）の観測。完走していない
- 当時ここから「ラウンド所要は最も遅い逐次チェーンで決まる／GPU を増やしても縮まない」と
  結論していたが、**これは誤り**だった。薄かったのは env actor が直列だったためで、
  ワークロードの性質ではない。宣言を入れた後の同じ構成は GPU を使い切る

**参加率もモデル規模で分かれる**

- 16 エージェント / 4 ラウンド / プール 400 で各 4 ラン

  | モデル | 参加率 |
  | --- | --- |
  | `Qwen3.6-35B-A3B-FP8` | 98 / 97 / 92 / 95 |
  | `Qwen3.5-4B` | 84 / 89 / 81 / 81 |

- 範囲が重ならない
- 4 エージェントでは分離しない（4B が 81 / 81 / 88、35B が 88 / 88 / 94 で 88 が重なる）
  - ➜ 頭数が増えるほどモデル能力の差が出る

**テンプレートキャッシュは embedding が無いと一度も効かない。あっても全部は効かない**

- `ask_env` は指示文から Python を生成して実行する。`_lookup` が過去のテンプレートに当てれば生成を省ける
- embedding が無いと `_lookup` は無条件で miss（`reason=embedding_unavailable`）。文字列一致のフォールバックは無い
- 実測（4B / 128 体、同一コードで `ENABLE_EMBEDDING` だけを変えた対照）

  | | キャッシュミス | embedding 失敗 | `ask_env` |
  | --- | --- | --- | --- |
  | 有り | 188 | 0 | 1549 回 / **6210 秒** |
  | 無し | 1540 | 3074 | 1693 回 / **10971 秒** |

- **規模が小さいと効かない。** 16 体では有り 1063〜1228 秒 / 無し 1125〜1574 秒で範囲が重なる
  - 同一テンプレートが繰り返される量が効くので、頭数が増えるほど差が開く
- **GPU は埋まらなかった。** 同時実行の中央値は 2 → 3、`Running>=8` は 3% → 7%
  - これも `max_concurrency=1` での観測。直列の待ち行列が短くなるだけで並列度は上がらない
  - embedding が効くのはキャッシュのヒット率であって、並列度ではない。この切り分けは今も有効

**残るミスは類似度の閾値で落ちている**

- embedding を立てると理由が `embedding_unavailable` → `below_similarity_threshold` に変わる
- env 型も変数キーも一致する候補がキャッシュにあるのに、**文字が同一の指示文が弾かれる**
- `CodeGenRouter` の `template_cache_similarity_threshold` は既定 **0.85**
  - `EnvRouterActor` はこれを渡さずに構築するので既定が固定される。環境変数も無い
- ➜ 128 体で 188 件のミスが残るのはこれ。テンプレートの種類数から予想される数より桁が多い
- 下げると減る。16 体・**各ラン冷たいキャッシュ**での実測

  | 閾値 | ミス |
  | --- | --- |
  | 0.85（上流の既定） | 27, 31 |
  | 0.60 | 6 |

  - 0.60 は 1 ラン ➜ 方向は一貫するが確定ではない
  - `EnvRouterActor` は `codegen_kwargs` を受け取る口を持つのに、この値を載せていない

**`EnvRouterActor` が 3 つの値をハードコードしている**

`ask_env` の速さと成否を決める 3 つが、いずれも設定から届かない。

| 値 | 既定 | 届かない理由 |
| --- | --- | --- |
| ルータ実装 | `CodeGenRouter` | `env_router_actor.py` が直接 import |
| `max_concurrency` | 8（既定）だが実効 1 | 全 env モジュールが `is_concurrency_safe()` を宣言しない限り 1 に落ちる |
| `template_cache_similarity_threshold` | 0.85 | `codegen_kwargs` に載っていない |

`EnvRouterProxy` は完成済みの actor ハンドルを受け取るだけなので、外から差し替える口も無い。

**env actor の直列化。宣言 1 行が、GPU が埋まるかどうかを決める**

`ENV_ACTOR_MAX_CONCURRENCY` の既定は 8 だが、`society/cli.py` が全 env モジュールの宣言を見る。

```python
max_concurrency = Config.ENV_ACTOR_MAX_CONCURRENCY if all_safe else 1
```

`EnvBase.is_concurrency_safe()` の既定は `False`。contrib の 16 モジュールのうち宣言しているのは
`GlobalInformationEnv` と `MobilitySpace` の 2 つだけなので、**それ以外を 1 つでも積むと 1 に落ちる**。

- 16 エージェント・4 ラウンド・各 4 ラン

  | | 参加率（中央値） | 1 ラウンド |
  | --- | --- | --- |
  | `max_concurrency=1` | 48% | 305 秒 |
  | `max_concurrency=8` | **81%** | **160 秒** |

- 1 では後半のラウンドで**提出がゼロ**になる。60 秒のリクエストタイムアウトを
  直列の待ち行列の後ろで使い切るため
- 35B・128 エージェント・`SocialMediaSpace`・宣言あり 2 本 / なし 2 本

  | | 同時リクエスト中央値 | 生成スループット中央値 |
  | --- | --- | --- |
  | 宣言あり | **17〜19** | **1322 tok/s** |
  | 宣言なし | 1〜2 | 343 tok/s |

  - 範囲が重ならない。クライアント側 AIMD セマフォは 180 まで開いた
  - この条件では SM 使用率 93〜94%・消費電力 467 W。GPU は使い切れる
- **宣言していないだけのモジュールが多い。** tool 本体に `await` を 1 つも持たないのは
  16 個中 14 個で、宣言済みの 2 個より強い条件を満たしている。`EnvRouterActor.ask` は
  `async def` なので Ray は単一イベントループで動かし、`await` の無い tool 本体は
  他の tool と交互実行されない
- 宣言は 2 つを同時に有効にする。**actor の並列度**と、`router_codegen` の
  `_execute_lock`（`_exec_lock_ctx()` が `nullcontext()` になる）
  - 後者は生成コードが**複数 tool をまたぐ区間**の直列化を外す。tool 1 つ 1 つが原子的でも、
    `get` → 判断 → `submit` の間に他エージェントが割り込めるようになる
  - ➜ 宣言する前に、そのモジュールが「エージェント自身のキーしか書かないか」を確認する

**代替ルータは動く。** `ReActRouter` は `env_benchmark` 用に見えるが、16 体で完走した。

| ルータ | 参加率 | `ask_env` | キャッシュミス |
| --- | --- | --- | --- |
| `codegen` | 12%, 83% | 659〜1222 秒 | 26, 17 |
| `react` | 86%, 78% | 891〜999 秒 | 0（そもそも使わない） |

- エラーは 0 件。`ask()` の署名は両者同一で、`RouterBase` が両方の引数を受け取る
  （`ReActRouter` が `super().__init__()` に転送していないだけ）
- **速くはならなかった。** コード生成の往復は消えるが、function calling の往復が入る
- 各 2 ラン。範囲が重なるので優劣は言えない

**1 回の行動に 8 万トークン。内訳は prefill が生成の 8 倍**

所要時間を見積もる単位は「何体か」ではなく「1 イベントに何トークンかかるか」。

- `Qwen/Qwen3.6-27B`（dense）・1 GPU・128 体・`SocialMediaSpace`・thinking 有効
- vLLM のスループットログ 4.0 時間分を積分し、`social_media_event` の件数で割った

  | | 合計 | 1 イベントあたり |
  | --- | --- | --- |
  | prefill | 32.9M tok | 約 **74,000** |
  | 生成 | 4.0M tok | 約 **9,000** |

- **律速は prefill。** 思考が長いことではなく、1 回の行動のために 7 万トークンの
  プロンプトを読ませていることが効く。ReAct ループが毎回ふくらんだ文脈を送り直すため
- ➜ 生成長だけを縛っても全体の 11% にしか効かない
- ➜ 減らすなら prefix caching のヒット率を見る（`--enable-prefix-caching` は既定で有効）
- n=1・完走していないランからの観測。桁は信用してよいが、係数は確定ではない

**`router_codegen.py` は確率的に init を落とす**

- `EnvRouterActor.init()` が統計コードを LLM に生成させ、実行して検証する
- 生成コードがリトライ後も実行に失敗すると、シミュレーション本体に入る前に落ちる

```
router_codegen.py:1888 _generate_statistics_code
ValueError: Generated statistics code failed execution after retries.
```

- 4B での実測: 同一構成で 1 回失敗・再実行で成功。再現性はない
- 確保した GPU 時間を起動だけで捨てることになる（実測 7 分）
- ➜ 再実行する。落ちたら構成を疑う前に、もう一度投げて切り分ける

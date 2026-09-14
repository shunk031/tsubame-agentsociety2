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
- **GPU は律速ではない。** 16 体で `Running` は平均 1〜3 本（容量は 32 並列・`max_num_seqs` 256）
  - ラウンド頭に全員が発火したあと脱同期する。ReAct ループが逐次なので、各エージェントは
    セマフォではなく自分の直前の呼び出しを待っている
  - ➜ ラウンド所要は**最も遅い逐次チェーン**で決まる。GPU を増やしても縮まない
- **128 体でも埋まらない。** 1145 サンプル中 `Waiting` がゼロでないのは 2 回だけ

  | 体数 | `Running` 中央値 | 最大 | `Waiting`>0 | KV 最大 |
  | --- | --- | --- | --- | --- |
  | 16 | 1〜3 | — | 0 | 0.7% |
  | 128 | **2** | 48 | 2 / 1113 | 13.8% |
  | 128（embedding 有り） | **3** | 52 | 1 / 664 | 15.4% |
  | 256 | **2** | 80 | 0 / 480 | 19.3% |

  - ➜ 頭数を 16 倍にしても飽和しない。最大値だけが伸びて定常は薄いまま
  - 256 体の行は wall clock で打ち切られたラン（1 ラウンドのみ）の観測。完走していない
  - ➜ GPU を増やす判断には、まず飽和させる方法が要る

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
- **GPU は埋まらない。** 同時実行の中央値は 2 → 3、`Running>=8` は 3% → 7%
  - 直列の待ち行列が短くなるだけで、並列度は上がらない

**残るミスは類似度の閾値で落ちている**

- embedding を立てると理由が `embedding_unavailable` → `below_similarity_threshold` に変わる
- env 型も変数キーも一致する候補がキャッシュにあるのに、**文字が同一の指示文が弾かれる**
- `CodeGenRouter` の `template_cache_similarity_threshold` は既定 **0.85**
  - `EnvRouterActor` はこれを渡さずに構築するので既定が固定される。環境変数も無い
- ➜ 128 体で 188 件のミスが残るのはこれ。テンプレートの種類数から予想される数より桁が多い

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

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

**適応セマフォが thinking のばらつきを輻輳と誤読して、並列度を自分で絞る**

- `config/llm_dispatcher.py` の `AdaptiveSemaphore`（AIMD）が LLM 呼び出しを挟む。プロセスごとに 1 個
- 既定値: 初期 `AGENTSOCIETY_LLM_RAY_CONCURRENCY=16` / `AGENTSOCIETY_LLM_LATENCY_DEGRADE_FACTOR=4.0` /
  `overload_threshold=0.1`（ハードコード）/ `AGENTSOCIETY_LLM_REQUEST_TIMEOUT=60`
- 判定は「ベースラインの 4 倍を超えたら slow」。1 ラウンドの 10% が slow なら limit を半減
- thinking 有効時の応答は 3〜37 秒で**固有に 12 倍ばらつく** ➜ 正常なばらつきが恒常的に slow 判定になる
  - レート制限は一度も起きない（ログは常に `429=0/N`）
  - 専用のローカルバックエンドに対しては、輻輳ではなく分散を測っているだけ
- 実測（4B / 16 エージェント / 4 ラウンド）

  | | 既定 | `LATENCY_DEGRADE_FACTOR=inf` |
  | --- | --- | --- |
  | `DECREASE` | 21 回以上 | **0 回** |
  | ReAct 失敗 | 48 件 | 12 件 |
  | ラウンド所要 | 429 / 224 / 299 / 60 分超 | 368 / 348 / 321 / 322 |

- ➜ ローカル vLLM では `AGENTSOCIETY_LLM_LATENCY_DEGRADE_FACTOR=inf` で相対判定を切る
  - コード側に `!= float("inf")` という無効化の分岐がある
- **GPU は律速ではない。** `Running` は平均 1〜3 本（容量は 32 並列・`max_num_seqs` 256）
  - ラウンド頭に全員が発火したあと脱同期する。ReAct ループが逐次なので、各エージェントは
    セマフォではなく自分の直前の呼び出しを待っている
  - ➜ ラウンド所要は**最も遅い逐次チェーン**で決まる。GPU を増やしても縮まない

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

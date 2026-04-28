## 📋 各メソッド概要（LangChain / LangGraph重点）

### 1. RAG全体フロー
```text
📄 Load → ✂️ Split → 🔢 Embed → 🗄️ Store

通常RAG
                                  ↓
                           質問入力(question)
                                  ↓
                     LangGraph: rewrite_query
                                  ↓
                           検索用質問(search_query)
                                  ↓
                            retrieve → context
                                  ↓
                            generate → answer

Function Calling RAG
                                  ↓
                           質問入力(question)
                                  ↓
                            agent(LLM)
                                  ↓
                tool_execution ← tool_calls の有無を判定
                     ↓                            ↑
 search_documents_tool / summarize_history_tool   │
                     ↓                            │
                            finalize ←───────────┘

Workflow Routing RAG
                                  ↓
                           質問入力(question)
                                  ↓
                classify_workflow_route で分類
                 ┌──────────────┼──────────────┐
                 ↓              ↓              ↓
            document          web          general
                 ↓              ↓              ↓
 workflow_document_     workflow_web_   workflow_general_
 rewrite_query          search          answer
                 ↓              ↓
 workflow_document_   workflow_generate_
 retrieve             web_answer
                 ↓
 workflow_generate_
 document_answer
```

このアプリでは、会話履歴機能を `st.session_state` による単純な履歴保持だけでなく、**LangGraph の状態遷移** を使って実装しています。
特に通常RAGでは、曖昧な follow-up 質問をそのまま検索せず、`rewrite_query_node()` で **検索向けの具体的な質問文へ補完** してから Retrieve します。
加えて、Function Calling RAG では、LLM が必要に応じてツールを選択し、`search_documents_tool()` や `summarize_history_tool()` の結果を使って最終回答を作る流れを学べます。
さらに、LLMルーティング RAG では、`classify_workflow_route_with_llm()` によって質問を `document / web / general` に分類し、質問内容に応じて **文書検索・Web向けフロー・通常応答フロー** を切り替える構成も学べます。

### 2. 補助関数・ノード一覧
| 関数名 | 役割 | 主なコンポーネント | フェーズ |
|--------|------|-------------------|----------|
| `check_api_key()` | APIキー確認 | - | 前処理 |
| `read_uploaded_file()` | txt / md を `Document` に変換 | `Document` | Load |
| `split_documents()` | 分割方式に応じて文書をチャンク化 | `RecursiveCharacterTextSplitter`, `CharacterTextSplitter` | Split |
| `build_vectorstore()` | 分割 → 埋め込み → Chroma保存 | `OpenAIEmbeddings`, `Chroma` | Split + Store |
| `format_chat_history()` | 直近の会話履歴を整形 | - | History整形 |
| `get_or_create_conversation_memory()` | `ConversationSummaryMemory` を初期化 | `ConversationSummaryMemory`, `ChatOpenAI` | History要約 |
| `rebuild_conversation_memory_from_history()` | 通常履歴から要約メモリを再構築 | `ConversationSummaryMemory` | History要約 |
| `format_chat_history_with_summary()` | 要約 + 直近ターンで履歴を整形 | `ConversationSummaryMemory` | History要約 |
| `get_prompt_template()` | プロンプト切り替え | `ChatPromptTemplate` | Prompt |
| `rewrite_query_node()` | 会話履歴を使って検索用クエリへ補完 | `ChatPromptTemplate`, `ChatOpenAI` | Query Rewrite |
| `retrieve_node()` | 補完後クエリで類似検索 | `Chroma` | Retrieve |
| `generate_node()` | 会話履歴 + 検索文脈で回答生成 | `ChatPromptTemplate`, `ChatOpenAI` | Generate |
| `build_rag_graph()` | LangGraph の実行フロー構築 | `StateGraph` | Orchestration |
| `answer_question()` | 初期状態を渡して LangGraph 実行 | `CompiledStateGraph` | Execute |
| `search_documents_tool()` | ベクトルDBを検索して根拠テキストを返す | `@tool`, `Chroma` | Tool Calling |
| `summarize_history_tool()` | 直近の会話履歴を整形して返す | `@tool` | Tool Calling |
| `tool_calling_llm_node()` | LLMが必要に応じてツール呼び出しを判断 | `ChatOpenAI.bind_tools` | Agent |
| `tool_execution_node()` | LLMが要求したツールを実行 | `ToolMessage` | Tool Execution |
| `tool_calling_finalize_node()` | 最終回答を state に格納 | - | Finalize |
| `should_continue_tool_calling()` | Tool Calling継続可否を判定 | - | Routing |
| `build_tool_calling_graph()` | Function Calling 用 LangGraph を構築 | `StateGraph` | Orchestration |
| `answer_question_with_tool_calling()` | Function Calling RAG を実行 | `CompiledStateGraph` | Execute |

### 3. LangGraph 化した会話履歴機能

#### `RAGState` - 状態定義
```python
class RAGState(TypedDict):
    question: str
    k: int
    prompt_type: str
    chat_history: list
    context_text: str
    retrieved_results: list
    answer: str
    search_query: str
```

- LangGraph で受け渡す状態を1つにまとめた型です。
- `question` はユーザーの元の質問、`search_query` は検索用に補完された質問です。
- `chat_history` は直近の会話文脈、`context_text` は検索結果をまとめた文脈です。
- `retrieved_results` と `answer` も状態として持つため、各ノードの責務が明確になります。

#### `rewrite_query_node()` - 会話履歴を使った質問補完
```python
def rewrite_query_node(state: RAGState):
    if not state["chat_history"]:
        return {"search_query": state["question"]}
```

```python
prompt = ChatPromptTemplate.from_template("""
あなたは検索用クエリ補完アシスタントです。
会話履歴を参考にして、現在の質問が曖昧なら意味が通る具体的な質問文へ補完してください。
明確な質問ならそのまま返してください。
""")
```

- follow-up 質問が曖昧なときに、会話履歴を見て検索向きの質問へ言い換えるノードです。
- たとえば「その根拠は？」だけでは検索に弱いため、直前の質問を踏まえて「リモートワークは何日まで可能かの根拠は？」のような形へ補完します。
- 会話履歴が空なら、元の `question` をそのまま `search_query` に使います。
- これにより、**履歴を回答生成だけでなく検索前段にも反映** できます。

#### `retrieve_node()` - 補完済みクエリで検索
```python
retrieved_results = vectorstore.similarity_search_with_score(
    state["search_query"],
    k=state["k"]
)
```

- 従来の実装では `question` そのもので検索していました。
- LangGraph 化後は `rewrite_query_node()` が作った `search_query` で検索します。
- この変更により、曖昧な省略質問でも Retrieve の精度を上げやすくなります。

#### `generate_node()` - 会話履歴 + 文脈で回答生成
```python
response = chain.invoke({
    "chat_history": history_text,
    "context": state["context_text"],
    "question": state["question"]
})
```

- 回答生成では、元の質問 `question` を維持したまま使います。
- 一方で、検索結果は `search_query` によって補完済みなので、文脈不足が起きにくくなります。
- これにより、自然な follow-up 質問を UI 上で保ちつつ、内部では検索しやすい形へ変換できます。

#### `build_rag_graph()` - 状態遷移の流れ
```python
graph_builder.add_node("rewrite_query", rewrite_query_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "rewrite_query")
graph_builder.add_edge("rewrite_query", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
```

- 会話履歴つきRAGの流れを、**質問補完 → 検索 → 回答生成** の3段階に分けています。
- LangGraph を使うことで、状態の流れとノード責務がコード上で見えやすくなります。
- 今後、要約ノードや履歴圧縮ノードを追加したい場合も、この構造にノードを足して拡張しやすいです。

### 3.5 ConversationSummaryMemory による履歴圧縮

#### 背景
- 直近数ターンだけを `format_chat_history()` で渡す方式はシンプルですが、会話が長くなると重要な前提が落ちやすくなります。
- 一方で、全履歴を毎回そのまま渡すと、トークン数が増えてコストやレイテンシが悪化しやすくなります。
- そのため今回の `app.py` では、`ConversationSummaryMemory` を使って**過去会話を要約しつつ、直近ターンは生のまま残す**構成を追加しています。

#### `SUMMARY_PROMPT_JA` - 日本語要約プロンプト
```python
SUMMARY_PROMPT_JA = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
あなたは会話履歴を日本語で要約するアシスタントです。
これまでの要約と新しい会話履歴をもとに、要点だけを自然な日本語で更新要約してください。
必ず日本語で出力してください。
""".strip(),
)
```

- `ConversationSummaryMemory` の既定動作に任せると、要約が英語で出る場合があります。
- そこで `SUMMARY_PROMPT_JA` を明示し、要約文を必ず日本語で更新するようにしています。

#### `get_or_create_conversation_memory()` - 要約メモリの生成
```python
def get_or_create_conversation_memory():
```

- Session State 上に `conversation_memory` がなければ、新しく `ConversationSummaryMemory` を作成します。
- 要約用LLMには `ChatOpenAI(model="gpt-4o-mini", temperature=0)` を使い、毎回ぶれにくい要約になるようにしています。

#### `rebuild_conversation_memory_from_history()` - 履歴からの再構築
```python
def rebuild_conversation_memory_from_history(chat_history):
```

- Streamlit は操作のたびに再実行されるため、通常履歴と要約メモリの整合性が崩れないよう、必要時に `chat_history` から再構築できるようにしています。
- 各ターンを `memory.save_context({"input": ...}, {"output": ...})` で順番に流し込み、要約を再生成します。

#### `format_chat_history_with_summary()` - 要約 + 直近ターン整形
```python
def format_chat_history_with_summary(chat_history, max_turns=3):
```

- この関数は、長い会話履歴をそのまま全部渡す代わりに、`[これまでの会話要約]` と `[直近の会話]` をまとめて返します。
- `memory_summary_enabled` がオフなら、従来どおり `format_chat_history()` を使います。
- `len(chat_history) <= max_turns` の間は無理に要約を前面に出さず、短い会話では直近履歴だけを返すようにしています。

#### どこで使われるか
- `classify_workflow_route_with_llm()` では、ルーティング判定時にも要約付き履歴を参照します。
- `rewrite_query_node()` では、曖昧な follow-up 質問を補完する際に要約付き履歴を参照します。
- `generate_node()` や `general_answer_node_response()` でも、回答生成時の会話文脈として要約付き履歴を使います。
- `summarize_history_tool()` も `format_chat_history_with_summary()` を返すため、Function Calling RAG でも長い履歴を圧縮した形で扱えます。

#### UIでの確認ポイント
- `ConversationSummaryMemoryで長い履歴を要約する` のチェックボックスで有効/無効を切り替えられます。
- `要約とは別に保持する直近ターン数` のスライダーで、何ターンを生のまま残すか調整できます。
- `render_chat_history_view()` 内では `ConversationSummaryMemory の要約結果` を展開表示できるため、内部の圧縮結果を画面上で確認できます。
- `reset_chat_history()` では通常履歴だけでなく `conversation_memory.clear()` も行い、表示履歴と内部要約のずれを防いでいます。

### 4. なぜ LangGraph を使うのか

従来の `st.session_state` ベース実装では、会話履歴は保持できても、検索前処理・検索・回答生成が1つの関数にまとまりやすく、処理の責務が見えにくくなります。
LangGraph を使うと、**どの段階で会話履歴を使うか** をノード単位で整理できます。

特に今回の実装では、会話履歴の使いどころが2つあります。

- `rewrite_query_node()` で検索前に質問を補完する。
- `generate_node()` で回答生成時に文脈補完として使う。

この2段構えにすることで、会話履歴ありの効果が出やすくなります。

### 5. 旧実装との違い

#### 旧実装
```text
question → similarity_search_with_score(question) → context → LLM
                           ↑
                    chat_history は回答時だけ参照
```

#### 現実装
```text
question + chat_history
        ↓
rewrite_query_node
        ↓
search_query
        ↓
similarity_search_with_score(search_query)
        ↓
context + chat_history + question
        ↓
generate_node
        ↓
answer
```

- 旧実装では、会話履歴は主に回答生成時の補助でした。
- 現実装では、会話履歴を検索前にも活用しています。
- そのため、「その根拠は？」「その申請方法は？」のような省略質問への耐性が上がります。

### 6. 会話履歴あり・なしで何が変わるか

#### 会話履歴OFF
- `chat_history` は空で渡されます。
- `rewrite_query_node()` は `question` をそのまま `search_query` に使います。
- 単発RAGとして動作します。

#### 会話履歴ON
- 直近の質問と回答が `chat_history` に入ります。
- `rewrite_query_node()` がそれを参照して質問を具体化します。
- `generate_node()` も会話文脈を見ながら回答します。

### 7. Function Calling（Tool Calling）の構成

#### `ToolCallingState` - Function Calling用の状態定義
```python
class ToolCallingState(TypedDict):
    question: str
    prompt_type: str
    chat_history: list
    answer: str
    tool_trace: list
    messages: list
```

- `messages` を state に持たせることで、`AIMessage` と `ToolMessage` の往復を LangGraph 上で追跡できます。
- `tool_trace` は UI の Tool Callingログ表示に使い、どのツールが呼ばれたかを確認するための状態です。

#### `search_documents_tool()` - 文書検索ツール
```python
@tool
def search_documents_tool(query: str) -> str:
```

- ベクトルDBに対して類似検索を行い、関連チャンクを根拠テキストとして返します。
- Function Calling RAG では、LLM が「検索が必要」と判断したときにこのツールを呼び出します。
- 実行結果は `last_retrieved_results` にも反映されるため、通常RAGと同じように検索根拠を UI で確認できます。

#### `summarize_history_tool()` - 会話履歴要約ツール
```python
@tool
def summarize_history_tool() -> str:
```

- 直近の会話履歴を短く整形して返します。
- 省略された follow-up 質問に対し、LLM が履歴要約を必要と判断した場合に使えます。

#### `tool_calling_llm_node()` - ツール呼び出し判断
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
```

- `bind_tools(TOOLS)` によって、LLM が `search_documents_tool()` と `summarize_history_tool()` を選択できるようにしています。
- system prompt では、ツール結果を根拠に回答すること、根拠不足なら「わかりません」と答えることを明示しています。

#### `tool_execution_node()` - ツール実行
```python
for tool_call in getattr(last_message, "tool_calls", []):
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})
    tool_result = tool_map[tool_name].invoke(tool_args)
```

- LLM が返した `tool_calls` を受け取り、対応する Python関数を実行します。
- 実行結果は `ToolMessage` として `messages` に追加し、再び LLM へ戻します。

#### `should_continue_tool_calling()` - 条件分岐
```python
if getattr(last_message, "tool_calls", None):
    return "tool_execution"
return "finalize"
```

- LLM がツール呼び出しを返した場合は `tool_execution` へ進みます。
- 返していない場合は、そのまま `finalize` に進んで最終回答を確定します。

#### `build_tool_calling_graph()` - 状態遷移
```python
graph_builder.add_node("agent", tool_calling_llm_node)
graph_builder.add_node("tool_execution", tool_execution_node)
graph_builder.add_node("finalize", tool_calling_finalize_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue_tool_calling,
    {
        "tool_execution": "tool_execution",
        "finalize": "finalize",
    },
)
graph_builder.add_edge("tool_execution", "agent")
graph_builder.add_edge("finalize", END)
```

- `add_node()` は処理ノードの登録、`add_edge(A, B)` は「Aの次にBへ進む」という固定ルート定義です。
- `add_edge(START, "agent")` は開始時に最初のノードとして `agent` を実行する、`add_edge("tool_execution", "agent")` はツール実行後にもう一度 LLM 判断へ戻す、`add_edge("finalize", END)` は最終回答後に終了する、という意味です。
- `agent` の次だけは固定ではないため `add_conditional_edges()` を使い、`should_continue_tool_calling()` が `"tool_execution"` を返せばツール実行へ、`"finalize"` を返せば終了処理へ進みます。
- そのため Function Calling RAG 全体は、`START -> agent -> (tool_execution -> agent を必要なだけ繰り返す) -> finalize -> END` という状態遷移になります。

```
### Function Calling のグラフ
START
  ↓
agent
  ├─ tool_callsあり → tool_execution
  │                     ↓
  │                   agent に戻る
  │
  └─ tool_callsなし → finalize
                        ↓
                       END
```

### 8. Workflow Routing RAG の構成

#### Workflow Routing RAG の全体フロー
Workflow Routing RAG では、最初に LLM が質問内容を見て、どの処理フローに進むべきかを判定します。
この実装では、質問を `document` / `web` / `general` の3種類に分類し、その結果に応じて後続のノードを切り替えます。

```text
question
   ↓
classify_workflow_route_with_llm()
   ↓
document / web / general
   ├─ document → workflow_document_rewrite_query_node()
   │              → workflow_document_retrieve_node()
   │              → workflow_generate_document_answer_node()
   │              → answer
   │
   ├─ web      → workflow_web_search_node()
   │              → workflow_generate_web_answer_node()
   │              → answer
   │
   └─ general  → workflow_general_answer_node()
                  → answer
```

- `document` は、アップロード済み文書の内容を検索すべき質問です。
- `web` は、外部情報や最新情報を使って答えるべき質問です。
- `general` は、検索を使わず通常の LLM 応答で十分な質問です。

この構成により、すべての質問を同じRAGフローで処理するのではなく、質問の種類ごとに最適な処理へ分岐できます。

#### `WorkflowRoutingState` - LLMルーティング用の状態定義
```python
class WorkflowRoutingState(TypedDict):
    question: str
    k: int
    prompt_type: str
    chat_history: list
    route: str
    context_text: str
    retrieved_results: list
    answer: str
    search_query: str
    persist_dir: str
    web_context: str
```

- `route` に分類結果を保持し、1つの state でルーティングから回答生成まで扱います。
- `web_context` を持つことで、web フローで作成した調査メモを UI に表示できます。
- `persist_dir` を state に含めることで、document フローでは既存の Chroma インデックスをそのまま利用できます。

#### `classify_workflow_route_with_llm()` - ルート分類
```python
def classify_workflow_route_with_llm(question, chat_history):
```

- 質問と会話履歴をもとに、`document / web / general` の3種類へ分類します。
- 社内文書・アップロード資料・マニュアル参照は `document`、最新情報や外部サービス情報は `web`、言い換えや相談は `general` というルールです。
- 想定外の出力が返った場合は `document` にフォールバックするため、後続の分岐が壊れにくくなっています。

#### `classify_workflow_route_node()` - 分岐起点ノード
```python
def classify_workflow_route_node(state: WorkflowRoutingState):
    route = classify_workflow_route_with_llm(state["question"], state["chat_history"])
    return {"route": route}
```

- LangGraph の最初のノードとして動き、分類結果を `state["route"]` に格納します。
- 後続の `add_conditional_edges()` は、この `route` を見て document / web / general の各経路へ分岐します。

#### document フロー
```text
classify_workflow_route
  → workflow_document_rewrite_query
  → workflow_document_retrieve
  → workflow_generate_document_answer
```

- 文書を参照すべき質問は、通常RAGと同様に質問補完 → ベクトル検索 → 回答生成の流れで処理します。
- LLMルーティングは通常RAGを置き換えるというより、通常RAGへ入る前段に「どの処理に進むかの判定」を追加した構成です。

#### web フロー
```text
classify_workflow_route
  → workflow_web_search
  → workflow_generate_web_answer
```

- `search_web_context()` で外部情報の調査メモを作り、その内容を文脈として回答を生成します。
- 現在の実装は学習用の擬似Webコンテキスト生成であり、将来的に検索APIへ差し替えやすい構成です。

#### general フロー
```text
classify_workflow_route
  → workflow_general_answer
```

- 言い換え、相談、概念説明など、検索なしで十分な質問はこの経路で処理します。
- 不要なベクトル検索を避けられるため、質問の種類に応じた挙動の違いを学びやすくなります。

#### `build_workflow_routing_graph()` - 状態遷移
```python
graph_builder.add_node("classify_workflow_route", classify_workflow_route_node)
graph_builder.add_node("workflow_document_rewrite_query", workflow_document_rewrite_query_node)
graph_builder.add_node("workflow_document_retrieve", workflow_document_retrieve_node)
graph_builder.add_node("workflow_generate_document_answer", workflow_generate_document_answer_node)
graph_builder.add_node("workflow_web_search", workflow_web_search_node)
graph_builder.add_node("workflow_generate_web_answer", workflow_generate_web_answer_node)
graph_builder.add_node("workflow_general_answer", workflow_general_answer_node)
```

```python
graph_builder.add_edge(START, "classify_workflow_route")
graph_builder.add_conditional_edges(
    "classify_workflow_route",
    decide_workflow_after_classification,
    {
        "workflow_document_rewrite_query": "workflow_document_rewrite_query",
        "workflow_web_search": "workflow_web_search",
        "workflow_general_answer": "workflow_general_answer",
    },
)
```

- `classify_workflow_route` の後だけが条件分岐になっており、分類結果に応じて3経路へ遷移します。
- document は `workflow_document_rewrite_query -> workflow_document_retrieve -> workflow_generate_document_answer -> END`、web は `workflow_web_search -> workflow_generate_web_answer -> END`、general は `workflow_general_answer -> END` です。


#### Function Calling との違い

| 項目 | Function Calling RAG | LLMルーティング RAG |
|---|---|---|
| LLMの主な役割 | 必要なツールを選んで呼び出す | 最適な処理フローを選ぶ |
| 中心となる分岐 | `tool_calls` の有無で分岐 | `route` の値で分岐 |
| 実行単位 | ツールを1つずつ呼び出して往復する | 最初に経路を1回決めて、その後は対応フローを進む |
| 代表ノード | `tool_calling_llm_node()`、`tool_execution_node()` | `classify_workflow_route_node()`、`decide_workflow_after_classification()` |
| 有効な場面 | 「まず文書検索するべきか」「履歴要約も必要か」など、質問ごとに必要な処理が細かく変わる場合 | 「この質問は文書検索か、Webか、通常回答か」を最初に大きく振り分けたい場合 |
| 具体例 | 社内規定QAで、質問によっては文書検索だけで足りるが、曖昧な follow-up では履歴要約も併用したい場合 | チャットアプリで、社内文書への質問・一般相談・最新情報確認が混在している場合 |
| 費用面の傾向 | ツール呼び出しの往復が増えるほど LLM 呼び出し回数も増えやすく、コストは高くなりやすい | 最初の分類コストは増えるが、不要な検索や不要なツール実行を避けやすく、全体コストを抑えやすい場合がある |
| レイテンシ | ツール呼び出しを繰り返すと遅くなりやすい | 最初の分類後は単純なフローに流れるため、比較的安定しやすい |
| 設計の向き不向き | 細かい判断を LLM に委ねたいが、挙動が複雑になりやすい | 大きな分岐を明示したいときに向くが、各ルートの設計は別途必要 |
| UI表示 | Tool Callingログを確認できる | 選択された処理フローと Web調査メモを確認できる |

- Function Calling は「どのツールを使うか」を LLM が判断する方式です。
- LLMルーティングは「どのワークフローに進むか」を LLM が判断する方式です。
- つまり、Function Calling はツール選択中心、LLMルーティングは処理経路選択中心という違いがあります。

### 9. 実行フロー

#### 通常RAG
1. 文書をアップロードして `build_vectorstore()` を実行する。
2. ユーザーが質問を入力する。
3. `answer_question()` が通常RAG用の `initial_state` を作る。
4. `build_rag_graph()` で作成した LangGraph を実行する。
5. `rewrite_query_node()` が会話履歴をもとに `search_query` を作る。
6. `retrieve_node()` が `search_query` を使ってベクトル検索し、`context_text` を作る。
7. `generate_node()` が会話履歴と検索結果を使って回答を生成する。
8. 回答と検索根拠を UI に表示する。

#### Function Calling RAG
1. 文書をアップロードしてベクトルDBを準備する。
2. ユーザーが質問を入力する。
3. `answer_question_with_tool_calling()` が Function Calling 用の state を作る。
4. `build_tool_calling_graph()` の `agent` ノードで、LLM がツール利用の要否を判定する。
5. 必要であれば `search_documents_tool()` や `summarize_history_tool()` を呼び出す。
6. `tool_execution_node()` がツールを実行し、その結果を `ToolMessage` として state に追加する。
7. LLM はツール結果を受けて、さらにツールを呼ぶか、そのまま最終回答に進むかを判断する。
8. `tool_calling_finalize_node()` が最終回答を確定し、UI に Tool Calling ログとあわせて表示する。

#### Workflow Routing RAG
1. 文書をアップロードしてベクトルDBを準備する。
2. ユーザーが質問を入力する。
3. `answer_question_with_workflow_routing()` が Workflow Routing 用の state を作る。
4. `build_workflow_routing_graph()` の `classify_workflow_route_node()` が、質問を `document / web / general` に分類する。
5. `document` の場合は、質問補完 → 文書検索 → 回答生成の順に実行する。
6. `web` の場合は、Web調査メモを生成し、その内容をもとに回答生成する。
7. `general` の場合は、検索を行わず通常の LLM 応答を返す。
8. UI には回答に加えて、選択されたルートや必要に応じて Web調査メモを表示する。

### 10. 学習価値

この実装で学べることは次の通りです。

- RAG の基本構成である Load / Split / Store / Retrieve / Generate。
- 会話履歴を LLM に渡すだけでは不十分で、**検索前の質問補完** が重要なこと。
- LangGraph によって、状態遷移ベースで RAG を組み立てられること。
- 将来的に Query Routing、History Compression、Tool Calling などへ拡張しやすい構成を作れること。

### 11. 重要パラメータ解説

#### temperature

LLMの回答のランダム性（創造性）を制御するパラメータ。

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

| 値 | 挙動 | アプリでの用途 |
|---|---|---|
| `0` | 毎回同じ回答（ぶれなし） | ルート分類・クエリ補完・RAG回答すべて |
| `0.3〜0.7` | 安定しつつ少し柔軟 | チャットボット、要約 |
| `1.0〜` | 多様でクリエイティブ | アイデア出し、物語生成 |

RAGでは根拠文書への**忠実度を上げるため `temperature=0` で固定**している。


#### `k` - 取得する文書数

- `k` は、ベクトル検索で取得する文書数を表します。
- `retrievenode()` では `similarity_search_with_score(..., k=state.k)` に使われ、検索結果の件数を直接左右します。
- 値を大きくすると参照文脈は増えますが、ノイズも増えやすくなります。
- 値を小さくすると文脈は絞られますが、必要な情報を取りこぼす可能性があります。

#### `prompttype` - プロンプトの種類

- `prompttype` は、回答生成に使うプロンプトテンプレートの種類を切り替えるパラメータです。
- `get_prompt_template(prompttype)` に渡され、用途に応じたプロンプトを選択します。
- たとえば、["初心者向け", "要約重視", "箇条書き重視"]のような、異なるテンプレートを使い分ける想定です。
- 適切な `prompttype` を選ぶことで、回答スタイルや制約条件を切り替えられます。


#### `chunksize` - 分割サイズ

- `chunksize` は、アップロードした文書を分割するときの1チャンクあたりの文字数を指定します。
- `split_documents()` で使われ、検索単位の粒度に影響します。
- 大きくすると1チャンクに含まれる文脈は増えますが、検索精度が下がる場合があります。
- 小さくすると検索の粒度は細かくなりますが、前後関係が途切れやすくなります。

#### `chunkoverlap` - 重なり幅

- `chunkoverlap` は、隣接するチャンク同士にどれだけ重複を持たせるかを指定します。
- `split_documents()` で使われ、チャンク境界で情報が切れないようにする役割があります。
- 値を大きくすると文脈のつながりは保ちやすくなりますが、重複データが増えます。
- 値を小さくすると重複は減りますが、前後の情報が分断されやすくなります。

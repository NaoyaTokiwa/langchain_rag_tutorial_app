## 📋 各メソッド概要（LangChain / LangGraph重点）

### 1. RAG全体フロー
```text
📄 Load → ✂️ Split → 🔢 Embed → 🗄️ Store
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
```

このアプリでは、会話履歴機能を `st.session_state` による単純な履歴保持だけでなく、**LangGraph の状態遷移** を使って実装しています。
特に、曖昧な follow-up 質問をそのまま検索せず、`rewrite_query_node()` で **検索向けの具体的な質問文へ補完** してから Retrieve するのが重要な変更点です。

### 2. 補助関数・ノード一覧
| 関数名 | 役割 | 主なコンポーネント | フェーズ |
|--------|------|-------------------|----------|
| `check_api_key()` | APIキー確認 | - | 前処理 |
| `read_uploaded_file()` | txt / md を `Document` に変換 | `Document` | Load |
| `split_documents()` | 分割方式に応じて文書をチャンク化 | `RecursiveCharacterTextSplitter`, `CharacterTextSplitter` | Split |
| `build_vectorstore()` | 分割 → 埋め込み → Chroma保存 | `OpenAIEmbeddings`, `Chroma` | Split + Store |
| `format_chat_history()` | 直近の会話履歴を整形 | - | History整形 |
| `get_prompt_template()` | プロンプト切り替え | `ChatPromptTemplate` | Prompt |
| `rewrite_query_node()` | 会話履歴を使って検索用クエリへ補完 | `ChatPromptTemplate`, `ChatOpenAI` | Query Rewrite |
| `retrieve_node()` | 補完後クエリで類似検索 | `Chroma` | Retrieve |
| `generate_node()` | 会話履歴 + 検索文脈で回答生成 | `ChatPromptTemplate`, `ChatOpenAI` | Generate |
| `build_rag_graph()` | LangGraph の実行フロー構築 | `StateGraph` | Orchestration |
| `answer_question()` | 初期状態を渡して LangGraph 実行 | `CompiledStateGraph` | Execute |

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

### 7. 実行フロー
1. 文書をアップロードして `build_vectorstore()` を実行する。
2. ユーザーが質問を入力する。
3. `answer_question()` が `initial_state` を作る。
4. `build_rag_graph()` で作った LangGraph を実行する。
5. `rewrite_query_node()` が `search_query` を作る。
6. `retrieve_node()` が `search_query` で検索し、`context_text` を作る。
7. `generate_node()` が会話履歴と検索結果を使って回答する。
8. 回答と検索根拠を UI に表示する。

### 8. 学習価値

この実装で学べることは次の通りです。

- RAG の基本構成である Load / Split / Store / Retrieve / Generate。
- 会話履歴を LLM に渡すだけでは不十分で、**検索前の質問補完** が重要なこと。
- LangGraph によって、状態遷移ベースで RAG を組み立てられること。
- 将来的に Query Routing、History Compression、Tool Calling などへ拡張しやすい構成を作れること。

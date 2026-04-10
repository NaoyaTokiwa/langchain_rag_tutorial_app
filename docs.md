## 📋 各メソッド概要（LangChain重点）

### 1. RAG全体フロー
```text
📄 Load → ✂️ Split → 🔢 Embed → 🗄️ Store → 🔍 Retrieve → 🧠 Prompt → 🤖 Generate
```

### 2. 補助関数一覧
| 関数名 | 役割 | LangChainコンポーネント | RAGフェーズ |
|--------|------|-------------------|-------------|
| `check_api_key()` | APIキー確認 | - | 前処理 |
| `read_uploaded_file()` | txt / md を `Document` に変換 | `Document` | Load |
| `build_vectorstore()` | 分割 → 埋め込み → Chroma保存 | `RecursiveCharacterTextSplitter`, `OpenAIEmbeddings`, `Chroma` | Split + Store |
| `get_prompt_template()` | プロンプト切り替え | `ChatPromptTemplate` | Prompt |
| `answer_question()` | 検索 → プロンプト適用 → LLM回答生成 | `Chroma`, `ChatPromptTemplate`, `ChatOpenAI`, `chain` | Retrieve + Generate |

### 3. LangChain重点解説

#### `read_uploaded_file()` - Loadフェーズ
```python
Document(page_content=text, metadata={"source": filename})
```

- 役割: StreamlitのアップロードファイルをLangChain標準の `Document` 形式に変換する
- `Document` は本文 `page_content` と出典情報 `metadata` を持つ
- `metadata` にファイル名を持たせることで、根拠表示やデバッグに使える
- RAGでは、後続処理が `Document` を前提に進む

#### `build_vectorstore()` - Split + Storeフェーズ
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory="chroma_db"
)
```

- `RecursiveCharacterTextSplitter` で長文を検索しやすいチャンクに分割する
- `OpenAIEmbeddings` で各チャンクをベクトル化する
- `Chroma` に保存することで、類似検索できる状態にする

**LangChainの基本パターン**
```text
RecursiveCharacterTextSplitter → OpenAIEmbeddings → Chroma.from_documents
       ↓                           ↓                        ↓
   文書分割 → 文章→ベクトル変換 → 類似検索可能DB作成
```

#### `get_prompt_template()` - Promptフェーズ
```python
def get_prompt_template(prompt_type):
    templates = {
        "初心者向け": "...",
        "要約重視": "...",
        "箇条書き重視": "..."
    }
    return ChatPromptTemplate.from_template(templates[prompt_type])
```

- 回答スタイルをプロンプトごとに切り替えるための関数
- `ChatPromptTemplate` をテンプレートごとに出し分ける
- 同じ検索結果でも、プロンプト次第で回答の形を変えられる
- 検索処理と回答スタイルを分けて設計できるのがポイント

#### `answer_question()` - Retrieve + Generateフェーズ
```python
retrieved_results = vectorstore.similarity_search_with_score(question, k=k)
retrieved_docs = [doc for doc, score in retrieved_results]
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = get_prompt_template(prompt_type)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm
response = chain.invoke({"context": context_text, "question": question})
```

- `similarity_search_with_score()` で、質問に近いチャンクをスコア付きで取得する
- 取得したチャンク本文を結合して `context_text` を作る
- `get_prompt_template(prompt_type)` で選択中のプロンプトを取得する
- `ChatOpenAI` に文脈と質問を渡して回答を生成する

**LangChainの直感的パイプライン**
```text
similarity_search_with_score → context_text → get_prompt_template → ChatOpenAI → 回答
             ↓                      ↓                ↓                ↓
      スコア付き検索 → 文脈化 → プロンプト切替 → LLM実行 → 自然言語回答
```

### 4. 検索件数 `k` の可変化
検索件数 `k` を可変にすると、何件の関連チャンクを取得するかをUIから調整できます。

```python
retrieved_results = vectorstore.similarity_search_with_score(question, k=k)
```

この機能により、Retriever設計のトレードオフを確認できます。

- `k=1` の場合: 根拠が少なく、回答が短くなりやすい
- `k=5` 以上の場合: 根拠は増えるが、関係の薄い情報も混ざりやすい

つまり、`k` は「情報量」と「ノイズ量」のバランスを見ながら調整する必要があります。

### 5. 類似度スコア表示
検索結果ごとにスコアを表示すると、「なぜそのチャンクが選ばれたのか」を確認できます。

```python
for doc, score in retrieved_results:
    print(score)
```

- スコアを見ることで、各チャンクの近さを比較できる
- Chroma の `similarity_search_with_score()` が返す値は一般に **distance（距離）**
- **値が小さいほど関連性が高い** と解釈する

この表示により、Retriever・Embedding・Chunk設計の影響を可視化できます。  
類似度スコアが良いチャンクほど、回答根拠として使われやすいことを確認できます(以下図参照)。

<p align="center">
  <img src="images/類似度確認結果.png" alt="類似度確認結果" width="900">
</p>

### 6. プロンプト切り替え機能
このアプリでは、UIからプロンプトタイプを選択できます。

```python
prompt_type = st.selectbox(
    "🧠 プロンプトタイプ",
    ["初心者向け", "要約重視", "箇条書き重視"]
)
```

- **初心者向け**: やさしく丁寧に説明する
- **要約重視**: 要点を短くまとめる
- **箇条書き重視**: ポイントを整理して列挙する

この機能により、同じ検索結果でもプロンプトで回答表現が変わることを確認できます。
指定したプロンプト通りの動作になっていることを確認できます（以下は箇条書き重視の出力結果例）。

<p align="center">
  <img src="images/箇条書き重視.png" alt="類似度確認結果" width="900">
</p>

### 7. LangChainコンポーネントマップ
| 機能 | LangChain部品 | 使用関数 |
|------|---------------|----------|
| 読み込み | `Document` | `read_uploaded_file()` |
| 分割 | `RecursiveCharacterTextSplitter` | `build_vectorstore()` |
| 埋め込み | `OpenAIEmbeddings` | `build_vectorstore()` |
| ベクトルDB | `Chroma` | `build_vectorstore()` |
| 検索 | `similarity_search_with_score()` | `answer_question()` |
| プロンプト切替 | `ChatPromptTemplate` | `get_prompt_template()` |
| LLM | `ChatOpenAI` | `answer_question()` |
| チェーン | `prompt \| llm` | `answer_question()` |

### 8. 実行フロー（UI連携）
1. 左カラムでファイルを選択し、`read_uploaded_file()` で読み込む
2. `build_vectorstore()` で分割・埋め込み・保存を行う
3. 右カラムで質問、検索件数 `k`、プロンプトタイプを指定する
4. `answer_question(question, k, prompt_type)` を実行する
5. 回答、検索根拠、類似度スコアを表示する

### 9. 学習価値
このコードでは、RAGとLangChainの標準的な構成を一通り学べます。

- Load → Split → Store（Indexing）
- Retrieve → Generate（クエリ処理）
- LangChainチェーン（`prompt | llm`）
- ベクトルDB永続化（`persist_directory`）
- UIでの検索根拠確認
- 検索件数 `k` によるRetriever設計の比較
- 類似度スコアによる検索品質の可視化
- プロンプト切り替えによる回答スタイルの比較

### 10. プロンプトの役割

#### 1. 役割定義
「あなたは初心者にもわかりやすく説明する親切なAIアシスタントです」

- 回答スタイルを「親切・初心者向け」に固定する
- トーンの一貫性を保つ

#### 2. ハルシネーション防止
```text
以下の参考文脈だけを使って質問に答えてください
```

- LLMの内部知識だけで推測回答するのを防ぐ
- 検索結果に基づく回答へ寄せる

#### 3. 正直さの強制
```text
文脈に答えがない場合は、「わかりません」と正直に伝えてください
```

- 根拠のない断定を避ける
- RAGの信頼性を高める

#### 4. 動的変数埋め込み
- `{context}` ← 検索結果
- `{question}` ← ユーザー質問

```python
chain.invoke({"context": context_text, "question": question})
```

#### 5. LangChainチェーン連携
```python
chain = prompt | llm
```

```text
テンプレート → LLM → 回答
    ↓
context と question を流し込んで自然言語回答を得る
```

### 11. プロンプト切り替えで学べること
プロンプト切り替え機能を入れると、Retriever は同じでも Prompt によって最終出力が変わることを体験できます。

たとえば同じ検索結果に対しても、次のように出力の形が変わります。

- **初心者向け**: 背景説明を含めてやさしく説明する
- **要約重視**: 結論を短く返す
- **箇条書き重視**: 情報を整理して見やすく返す

これは、検索設計と回答設計の役割分担を理解するうえで重要です。

### 12. 実際の動作フロー
```python
1. similarity_search_with_score(question, k=k) → retrieved_results
2. retrieved_docs = [doc for doc, score in retrieved_results]
3. context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
4. prompt = get_prompt_template(prompt_type)
5. chain.invoke({"context": context_text, "question": question})
```

```text
参考文脈: [検索結果]
質問: [ユーザー質問]
プロンプト: [選択中のスタイル]
↓
GPT → 回答
```

### 13. なぜこの設計か
```text
一般LLM → 推測回答しがち
↓
RAGプロンプト → 検索結果限定で回答
↓
さらにプロンプト切替 → 回答スタイルも制御可能
↓
信頼性向上 + 根拠明確化 + 表現調整
```

この設計は、RAGの信頼性を保ちながら、用途に応じて回答の見せ方まで制御するための基本形です。

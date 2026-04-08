### **1. RAG全体フロー**
📄 Load → ✂️ Split → 🔢 Embed → 🗄️ Store → 🔍 Retrieve → 🤖 Generate


### **2. 補助関数一覧**
| 関数名 | 役割 | LangChainコンポーネント | RAGフェーズ |
|--------|------|-------------------|-------------|
| `check_api_key()` | APIキー確認 | - | 前処理 |
| `read_uploaded_file()` | **txt/md→Document** | **`Document`** | **Load** |
| `build_vectorstore()` | **分割→埋め込み→Chroma** | **`TextSplitter`, `Embeddings`, `Chroma`** | **Split+Store** |
| `answer_question()` | **検索→LLM回答** | **`Retriever`, `PromptTemplate`, `ChatOpenAI`, `chain`** | **Retrieve+Generate** |

### **3. LangChain重点解説**

#### **`read_uploaded_file()` - Loadフェーズ**

```python
Document(page_content=text, metadata={"source": filename})
```
•	役割: Streamlitのファイル→LangChain標準 Document 形式変換  
•	 Document :  page_content (本文) +  metadata (出典情報)※  
※複数ファイルが入力されることが想定されるのでファイル名も使用
•	重要: RAGの全処理が Document 前提

#### ** `build_vectorstore()`  - Split+Storeフェーズ**
```python
# 1. 分割
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# 2. 埋め込み
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. ベクトルDB保存
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_db")

```

#### LangChainの基本パターン:
```
RecursiveCharacterTextSplitter → OpenAIEmbeddings → Chroma.from_documents
       ↓                           ↓                        ↓
文書分割 → 文章→ベクトル変換 → 類似検索可能DB作成
```
####  **`answer_question()`  - Retrieve+Generateフェーズ**
```python
# Retrieve: ベクトル検索
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(question)

# Generate: LLM回答
prompt = ChatPromptTemplate.from_template("文脈: {context}\n質問: {question}")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm  # LangChainチェーン（パイプ記法）
response = chain.invoke({"context": context_text, "question": question})
```
LangChainの直感的パイプライン:
```
retriever → context_text → ChatPromptTemplate → ChatOpenAI → 回答
      ↓           ↓                ↓                ↓
 検索    → 文脈化 → テンプレート → LLM → 自然言語回答
```

### ４. **LangChainコンポーネントマップ**
| 機能 | LangChain部品 | 使用関数 |
|------|---------------|----------|
| **読み込み** | `Document` | `read_uploaded_file()` |
| **分割** | `TextSplitter` | `build_vectorstore()` |
| **埋め込み** | `OpenAIEmbeddings` | `build_vectorstore()` |
| **ベクトルDB** | `Chroma` | `build_vectorstore()` |
| **検索** | `Retriever` | `answer_question()` |
| **プロンプト** | `ChatPromptTemplate` | `answer_question()` |
| **LLM** | `ChatOpenAI` | `answer_question()` |
| **チェーン** | `prompt \| llm` | `answer_question()` |

### ５. **実行フロー（UI連携）**
1. 左カラム: ファイル選択 → read_uploaded_file()
2. 設定調整 → build_vectorstore() → session_state.chunks保存
3. 右カラム: 質問入力 → answer_question()
4. 結果表示: 回答 + retrieved_docs（根拠確認）

### ６. **学習価値**
このコードはRAGの完全実装＋LangChain標準パターンを網羅：  
•	Load→Split→Store（Indexing）  
•	Retrieve→Generate（クエリ処理）  
•	LangChainチェーン（ prompt\|llm ）  
• ベクトルDB永続化（ persist_directory ）  
•	UIデバッグ可能（チャンク・根拠確認）  

### 7. プロンプトの5つの役割
#### 1. 役割定義（System指示）
「あなたは初心者にもわかりやすく説明する親切なAIアシスタントです」
•	効果: LLMに「親切・初心者向け」の回答スタイルを固定  
	•	重要: 一貫したトーン・品質を保証  
#### 2. ハルシネーション防止（最重要）
```
「以下の参考文脈だけを使って質問に答えてください」
```
❌ NG: 「LLMの全知識」で適当回答  
✅ OK: 「{context}の中身」だけ使用  
RAGの核心: 「検索結果外の情報は使わない」

#### 3. 正直さ強制
```
「文脈に答えがない場合は、『わかりません』と正直に伝えてください」
```
❌ 自信満々で嘘: 「〇〇です」（文脈にない）  
✅ 正直回答: 「わかりません」

#### 4. 動的変数埋め込み
{context} ← 検索結果（retrieved_docs）  
{question} ← ユーザー質問
実行時：
```python
chain.invoke({"context": "リモート週3日...", "question": "リモート何日？"})
↓
「参考文脈: リモート週3日...\n質問: リモート何日？」
```
#### 5. LangChainチェーン連携
```python
chain = prompt | llm  # パイプ記法
```
```
テンプレート → LLM → 回答
    ↓
{context, question} → GPT → 自然言語回答
```
#### 🔍 実際の動作フロー
```python
1. retriever.invoke(question) → retrieved_docs
2. context_text = "\n\n".join(doc.page_content)  
3. chain.invoke({"context": context_text, "question": question})
↓
「参考文脈: [検索結果]\n質問: [ユーザー質問]」→ GPT → 回答
```
#### 💡 なぜこの設計か
```
一般LLM → 「推測回答」しがち
↓
RAGプロンプト → 「検索結果限定」で正確回答
↓
信頼性UP + 根拠明確
```
このプロンプトは「RAGの信頼性を担保する設計」そのもの！

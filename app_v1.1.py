# ============================================
# 1. ライブラリの読み込み
# ============================================
# Streamlit: 画面UI作成用
# LangChain: RAG処理の部品群
# python-dotenv: .envファイル読み込み用

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv     # 🔑 .envファイルからAPIキー読み込み

# LangChain系（RAGの部品箱）
from langchain_core.documents import Document           # 📄 文書データ形式
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 🤖 OpenAI連携（回答生成・埋め込み）
from langchain_community.vectorstores import Chroma      # 🗄️ ベクトルデータベース
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✂️ 文書分割
from langchain.prompts import ChatPromptTemplate         # 💬 LLMへの指示文作成


# ============================================
# 2. 環境変数読み込み（最初に実行）
# ============================================
# .envファイルからOPENAI_API_KEYを読み込みます。
# これがないとOpenAI APIが使えません。
load_dotenv()


# ============================================
# 3. アプリ全体設定
# ============================================
# アプリの基本設定（タイトル、アイコン、レイアウト）を定義します。

APP_TITLE = "📚 LangChainで学ぶRAGチュートリアル"
PERSIST_DIR = Path("chroma_db")  # ベクトルDB保存フォルダ

st.set_page_config(
    page_title="LangChain RAG Tutorial", 
    page_icon="📚",
    layout="wide"
)

st.title(APP_TITLE)
st.caption("初心者向けに、RAGの流れを見える化したStreamlitアプリ")


# ============================================
# 4. Session State初期化（必須）
# ============================================
# Streamlitはボタン押すたびに上から再実行されるため、
# 重要な状態をsession_stateで保持します。
# 「まだ存在しないなら初期値を設定」の書き方です。

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False  # ベクトルDB作成済みフラグ

if "chunks" not in st.session_state:
    st.session_state.chunks = []  # 分割済み文書チャンク一覧

if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []  # 直近の検索結果

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""  # 直近の回答文


# ============================================
# 5. 補助関数：APIキー確認
# ============================================
def check_api_key():
    """OPENAI_API_KEYが設定されているかチェック"""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key)  # 空文字でないことを確認


# ============================================
# 6. 補助関数：ファイル読み込み（Load）
# ============================================
# RAGの最初のステップ「文書読み込み」です。
# Streamlitのfile_uploaderから受け取ったファイルを
# LangChainのDocument形式に変換します。

def read_uploaded_file(uploaded_file):
    """
    txt/mdファイルをLangChain Documentに変換
    
    Args:
        uploaded_file: st.file_uploader()の戻り値
    
    Returns:
        list[Document]: Documentオブジェクトのリスト
    """
    # ファイル拡張子をチェック（大文字小文字問わず）
    suffix = Path(uploaded_file.name).suffix.lower()
    
    if suffix in [".txt", ".md"]:
        # バイナリ→文字列変換（UTF-8）
        text = uploaded_file.getvalue().decode("utf-8")
        
        # Document形式に変換（page_content=本文、metadata=メタ情報）
        return [
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name}  # 出典情報
            )
        ]
    else:
        raise ValueError("対応ファイル: .txt, .md のみ")


# ============================================
# 7. 補助関数：ベクトルDB作成（Split + Store）
# ============================================
# RAGの前処理「文書分割→埋め込み→保存」です。
# LangChainの基本フロー「Load→Split→Store」に相当。

def build_vectorstore(documents, chunk_size, chunk_overlap):
    """
    文書を分割→埋め込み→Chroma保存
    
    Args:
        documents: 読み込んだDocumentリスト
        chunk_size: 1チャンクの文字数
        chunk_overlap: チャンク間の重なり文字数
    
    Returns:
        tuple: (vectorstore, 分割済みドキュメント)
    """
    # Step1: 文書分割（長い文章を小さく分割）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # 1チャンクの最大文字数
        chunk_overlap=chunk_overlap   # 隣のチャンクとの重なり
    )
    split_docs = splitter.split_documents(documents)
    
    # Step2: 前回のデータは削除（学習用に毎回作り直し）
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    
    # Step3: 埋め込みモデル作成（文章→数値ベクトル変換）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Step4: ChromaベクトルDB作成＆保存
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR)  # ディスク保存
    )
    
    return vectorstore, split_docs


# ============================================
# 8. 補助関数：RAG回答生成（Retrieve + Generate）
# ============================================
# RAGの本番処理「検索→回答生成」です。
# 「Retrieve→Generate」の流れを実装。

def answer_question(question):
    """
    RAGで質問に回答
    
    Args:
        question: ユーザーの質問文
    
    Returns:
        tuple: (回答文, 検索されたDocumentリスト)
    """
    # 保存済みベクトルDB読み込み
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
    # 検索設定（上位3件取得）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)  # 質問に近いチャンク検索
    
    # 検索結果を1つの文字列に結合（LLM用の参考文脈）
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # LLM指示テンプレート作成
    prompt = ChatPromptTemplate.from_template("""
あなたは初心者にもわかりやすく説明する親切なAIアシスタントです。
以下の参考文脈だけを使って質問に答えてください。
文脈に答えがない場合は、「わかりません」と正直に伝えてください。

参考文脈:
{context}

質問:
{question}
""")
    
    # LLM設定（回答の一貫性を高めるためtemperature=0）
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # チェーン作成（指示→LLM）
    chain = prompt | llm
    
    # 回答生成
    response = chain.invoke({
        "context": context_text,    # 検索結果
        "question": question        # ユーザーの質問
    })
    
    return response.content, retrieved_docs


# ============================================
# 9. アプリ説明エリア
# ============================================
with st.expander("📖 RAGとLangChainをやさしく理解する", expanded=True):
    st.markdown("""
### 🔍 RAGとは？
**Retrieval-Augmented Generation**（検索拡張生成）の略です。
- LLM単体 → 学習済み知識のみで回答
- RAG → **文書検索＋LLM回答**

### 💡 なぜ便利？
あなたの**社内資料・個人ファイル**をLLMに認識させられます。

### 🧩 LangChainとは？
LLMアプリを作るための**部品箱**です。
```
文書読み込み → 分割 → 埋め込み → 検索 → 回答生成
     ↓           ↓       ↓       ↓       ↓
LangChainが各工程をつなぎやすくします
```

### 🎯 このアプリで学べること
1. **文書アップロード**
2. **チャンク分割**（調整可能）
3. **ベクトルDB作成**
4. **質問検索**
5. **根拠付き回答**
    """)

st.markdown("---")


# ============================================
# 10. メインUI（左右2カラム）
# ============================================
left_col, right_col = st.columns([1, 1])


# ============================================
# 11. 左カラム：文書準備（Load→Split→Store）
# ============================================
with left_col:
    st.header("📁 文書準備")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "📄 テキストファイルを選択",
        type=["txt", "md"],
        help="学習用サンプル: data/sample_company_handbook.txt"
    )
    
    # 分割設定
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "📏 チャンクサイズ",
            300, 1500, 700, 100,
            help="1チャンクの文字数（大きい=文脈多め、小さい=細かい検索）"
        )
    with col2:
        chunk_overlap = st.slider(
            "🔗 重なり幅",
            0, 300, 100, 20,
            help="チャンク間の重なり（文脈切れを防ぐ）"
        )
    
    # インデックス作成ボタン
    if st.button("🚀 インデックス作成", use_container_width=True):
        if not uploaded_file:
            st.warning("❌ ファイルをアップロードしてください")
        elif not check_api_key():
            st.error("❌ .env に OPENAI_API_KEY を設定してください")
        else:
            try:
                with st.spinner("文書処理中...（初回は時間がかかります）"):
                    # 1. ファイル読み込み
                    docs = read_uploaded_file(uploaded_file)
                    
                    # 2. 分割＆ベクトルDB作成
                    _, split_docs = build_vectorstore(docs, chunk_size, chunk_overlap)
                    
                    # 3. 状態更新
                    st.session_state.update({
                        "vectorstore_ready": True,
                        "chunks": split_docs
                    })
                
                st.success(f"✅ 完了！チャンク数: **{len(split_docs)}**件")
                st.balloons()  # 成功エフェクト
                
            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")
                st.caption("ファイル形式やAPIキーを確認してください")
    
    # チャンク確認エリア
    if st.session_state.chunks:
        st.subheader("📋 分割結果")
        st.info(f"総チャンク数: **{len(st.session_state.chunks)}**件")
        
        # プレビュー件数選択
        preview_n = st.slider("プレビュー", 1, min(10, len(st.session_state.chunks)), 3)
        
        for i, doc in enumerate(st.session_state.chunks[:preview_n]):
            with st.expander(f"チャンク #{i+1}"):
                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                st.caption(f"📄 {doc.metadata.get('source', '不明')}")


# ============================================
# 12. 右カラム：質問＆回答（Retrieve→Generate）
# ============================================
with right_col:
    st.header("💬 質問する")
    
    # 質問入力
    question = st.text_input(
        "❓ 質問を入力",
        placeholder="例: 「リモートワークは何日まで可能？」",
        help="アップロードした文書に関する質問をどうぞ"
    )
    
    # 回答生成ボタン
    if st.button("🤖 回答生成", use_container_width=True):
        if not st.session_state.vectorstore_ready:
            st.warning("⚠️ まず「インデックス作成」を実行してください")
        elif not question.strip():
            st.warning("⚠️ 質問を入力してください")
        elif not check_api_key():
            st.error("❌ .env の OPENAI_API_KEY を確認")
        else:
            try:
                with st.spinner("検索中...回答生成中..."):
                    answer, retrieved_docs = answer_question(question)
                    
                    # 状態更新
                    st.session_state.update({
                        "last_answer": answer,
                        "last_retrieved_docs": retrieved_docs
                    })
                    
                st.success("✅ 回答完了！")
                
            except Exception as e:
                st.error(f"❌ 回答生成エラー: {str(e)}")
    
    # 回答表示
    if st.session_state.last_answer:
        st.subheader("📝 最終回答")
        st.markdown(f"**{st.session_state.last_answer}**")
    
    # 検索根拠表示
    if st.session_state.last_retrieved_docs:
        st.subheader("🔍 検索根拠（上位3件）")
        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            with st.expander(f"根拠 #{i+1}"):
                st.text(doc.page_content)
                st.caption(f"📄 {doc.metadata.get('source', '不明')}")


# ============================================
# 13. 学習ポイント
# ============================================
st.markdown("---")
with st.expander("🎓 学習のポイント", expanded=False):
    st.markdown("""
    ### 🔧 **調整パラメータ**
    - **chunk_size**: 大→文脈多め、小→細かい検索
    - **chunk_overlap**: 重なり多め→文脈連続性UP
    
    ### ✅ **RAGの流れ**
    ```
    文書 → 分割 → ベクトル化 → 保存
              ↓
    質問 → 検索 → 根拠取得 → LLM回答
    ```
    
    ### 💡 **確認すべき点**
    1. チャンク分割の具合
    2. 検索された根拠文書
    3. 回答の正確性
    """)

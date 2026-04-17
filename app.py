# ============================================
# ライブラリの読み込み
# ============================================
# Streamlit: 画面UI作成用
# LangChain: RAG処理の部品群
# python-dotenv: .envファイル読み込み用

import os
import shutil
from pathlib import Path
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv  # 🔑 .envファイルからAPIキー読み込み
from langchain.prompts import ChatPromptTemplate  # 💬 LLMへの指示文作成
from langchain_community.vectorstores import Chroma  # 🗄️ ベクトルデータベース

# LangChain系（RAGの部品箱）
from langchain_core.documents import Document  # 📄 文書データ形式
from langchain_openai import ChatOpenAI  # 🤖 OpenAI連携（回答生成・埋め込み）
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter  # ✂️ 文書分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph系（状態付き対話フロー）
from langgraph.graph import END, START, StateGraph

# ============================================
# 環境変数読み込み（最初に実行）
# ============================================
# .envファイルからOPENAI_API_KEYを読み込みます。
# これがないとOpenAI APIが使えません。
load_dotenv()

# ============================================
# アプリ全体設定
# ============================================
# アプリの基本設定（タイトル、アイコン、レイアウト）を定義します。

APP_TITLE = "📚 LangChainで学ぶRAGチュートリアル"
PERSIST_DIR = Path("chroma_db")  # ベクトルDB保存フォルダ
PROMPT_TYPES = ["初心者向け", "要約重視", "箇条書き重視"]
SPLITTER_TYPES = ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"]
SESSION_DEFAULTS = {
    "vectorstore_ready": False,
    "chunks": [],
    "last_retrieved_docs": [],
    "last_retrieved_results": [],
    "last_answer": "",
    "retrieval_k": 3,
    "prompt_type": "初心者向け",
    "splitter_type": "RecursiveCharacterTextSplitter",
    "compare_chunk_size": 1000,
    "compare_chunk_overlap": 200,
    "show_splitter_comparison": False,
    "chat_history": [],
    "use_chat_history": True,
    "rag_graph": None,
}

st.set_page_config(page_title="LangChain RAG Tutorial", page_icon="📚", layout="wide")

st.title(APP_TITLE)
st.caption("初心者向けに、RAGの流れを見える化したStreamlitアプリ")

# ============================================
# Session State初期化（必須）
# ============================================
# Streamlitはボタン押すたびに上から再実行されるため、
# 重要な状態をsession_stateで保持します。
# 「まだ存在しないなら初期値を設定」の書き方です。


def initialize_session_state():
    """Session Stateに初期値を設定"""
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================
# 補助関数：APIキー確認
# ============================================
def check_api_key():
    """OPENAI_API_KEYが設定されているかチェック"""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key)  # 空文字でないことを確認


# ============================================
# 補助関数：ファイル読み込み（Load）
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
                page_content=text, metadata={"source": uploaded_file.name}  # 出典情報
            )
        ]
    else:
        raise ValueError("対応ファイル: .txt, .md のみ")


# ============================================
# 補助関数：文書分割のみ実行
# ============================================
def split_documents(documents, splitter_type, chunk_size, chunk_overlap):
    """
    分割方式に応じて文書をチャンク分割する

    Args:
        documents: 読み込んだDocumentリスト
        splitter_type: 分割方式
        chunk_size: 1チャンクの文字数
        chunk_overlap: チャンク間の重なり文字数

    Returns:
        list[Document]: 分割済みDocumentリスト
    """
    if splitter_type == "RecursiveCharacterTextSplitter":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_type == "CharacterTextSplitter":
        splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"未対応の分割方式です: {splitter_type}")

    return splitter.split_documents(documents)


# ============================================
# 補助関数：ベクトルDB作成（Split + Store）
# ============================================
# RAGの前処理「文書分割→埋め込み→保存」です。
# LangChainの基本フロー「Load→Split→Store」に相当。


def build_vectorstore(
    documents, chunk_size, chunk_overlap, splitter_type="RecursiveCharacterTextSplitter"
):
    """
    文書を分割→埋め込み→Chroma保存

    Args:
        documents: 読み込んだDocumentリスト
        chunk_size: 1チャンクの文字数
        chunk_overlap: チャンク間の重なり文字数
        splitter_type: 分割方式

    Returns:
        tuple: (vectorstore, 分割済みドキュメント)
    """
    # Step1: 文書分割（長い文章を小さく分割）
    split_docs = split_documents(documents, splitter_type, chunk_size, chunk_overlap)

    # Step2: 前回のデータは削除（学習用に毎回作り直し）
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    # Step3: 埋め込みモデル作成（文章→数値ベクトル変換）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step4: ChromaベクトルDB作成＆保存
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),  # ディスク保存
    )

    return vectorstore, split_docs


# ============================================
# 補助関数：プロンプト取得
# ============================================
def get_prompt_template(prompt_type):
    """
    選択されたプロンプト種別に応じてテンプレートを返す

    Args:
        prompt_type: プロンプト種別

    Returns:
        ChatPromptTemplate
    """
    templates = {
        "初心者向け": """
あなたは初心者にもわかりやすく説明する親切なAIアシスタントです。
以下の参考文脈だけを主な根拠として使って質問に答えてください。
会話履歴は文脈補完のために参照して構いません。
文脈に答えがない場合は、「わかりません」と正直に伝えてください。

会話履歴:
{chat_history}

参考文脈:
{context}

質問:
{question}
""",
        "要約重視": """
あなたは要点を簡潔にまとめるアシスタントです。
以下の参考文脈だけを主な根拠として使って、質問への答えを3文以内で短く要約してください。
会話履歴は文脈補完のために参照して構いません。
文脈に答えがない場合は、「わかりません」と答えてください。

会話履歴:
{chat_history}

参考文脈:
{context}

質問:
{question}
""",
        "箇条書き重視": """
あなたは情報整理が得意なアシスタントです。
以下の参考文脈だけを主な根拠として使って質問に答えてください。
会話履歴は文脈補完のために参照して構いません。
回答は箇条書きで、重要ポイントを3〜5個に整理してください。
文脈に答えがない場合は、「わかりません」と答えてください。

会話履歴:
{chat_history}

参考文脈:
{context}

質問:
{question}
""",
    }

    return ChatPromptTemplate.from_template(templates[prompt_type])


# ============================================
# 補助関数：会話履歴整形
# ============================================
# 直近の会話履歴をLLMに渡しやすい文字列へ変換
def format_chat_history(chat_history, max_turns=3):
    """
    会話履歴をLLMに渡しやすい文字列に整形

    Args:
        chat_history: 質問と回答の履歴リスト
        max_turns: 参照する直近ターン数

    Returns:
        str: 整形済みの会話履歴テキスト
    """
    recent_history = chat_history[-max_turns:]

    if not recent_history:
        return "会話履歴なし"

    history_texts = []
    for i, turn in enumerate(recent_history, start=1):
        history_texts.append(
            f"[過去の会話 {i}]\n" f"質問: {turn['question']}" f"回答: {turn['answer']}"
        )

    return "\n\n".join(history_texts)


# ============================================
# LangGraph状態定義
# ============================================
class RAGState(TypedDict):
    question: str
    k: int
    prompt_type: str
    chat_history: list
    context_text: str
    retrieved_results: list
    answer: str
    search_query: str


# ============================================
# LangGraphノード：質問補完
# ============================================
def rewrite_query_node(state: RAGState):
    """
    会話履歴をもとに検索しやすい質問文へ補完する

    Args:
        state: LangGraphで扱う状態

    Returns:
        dict: 更新する状態
    """
    if not state["chat_history"]:
        return {"search_query": state["question"]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history_text = format_chat_history(state["chat_history"], max_turns=3)

    prompt = ChatPromptTemplate.from_template(
        """
あなたは検索用クエリ補完アシスタントです。
会話履歴を参考にして、現在の質問が曖昧なら意味が通る具体的な質問文へ補完してください。
明確な質問ならそのまま返してください。

会話履歴:
{chat_history}

現在の質問:
{question}
"""
    )

    chain = prompt | llm
    response = chain.invoke(
        {"chat_history": history_text, "question": state["question"]}
    )

    return {"search_query": response.content.strip()}


# ============================================
# LangGraphノード：検索
# ============================================
def retrieve_node(state: RAGState):
    """
    質問に関連する文書チャンクを検索する

    Args:
        state: LangGraphで扱う状態

    Returns:
        dict: 更新する状態
    """
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    retrieved_results = vectorstore.similarity_search_with_score(
        state["search_query"], k=state["k"]
    )

    retrieved_docs = [doc for doc, score in retrieved_results]
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    return {"retrieved_results": retrieved_results, "context_text": context_text}


# ============================================
# LangGraphノード：回答生成
# ============================================
def generate_node(state: RAGState):
    """
    検索結果と会話履歴を使って回答を生成する

    Args:
        state: LangGraphで扱う状態

    Returns:
        dict: 更新する状態
    """
    history_text = format_chat_history(state["chat_history"], max_turns=3)
    prompt = get_prompt_template(state["prompt_type"])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm

    response = chain.invoke(
        {
            "chat_history": history_text,
            "context": state["context_text"],
            "question": state["question"],
        }
    )

    return {"answer": response.content}


# ============================================
# 補助関数：LangGraph作成
# ============================================
def build_rag_graph():
    """
    検索→回答生成のLangGraphを作成する

    Returns:
        CompiledStateGraph: 実行可能なLangGraph
    """
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("rewrite_query", rewrite_query_node)

    graph_builder.add_edge(START, "rewrite_query")
    graph_builder.add_edge("rewrite_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()


# ============================================
# 補助関数：RAG回答生成（Retrieve + Generate）
# ============================================
# RAGの本番処理「検索→回答生成」です。
# 「Retrieve→Generate」の流れを実装。


def answer_question(question, k, prompt_type, chat_history=None):
    """
    LangGraphを使ってRAGで質問に回答

    Args:
        question: ユーザーの質問文
        k: 検索で取得するチャンク数
        prompt_type: 使用するプロンプト種別
        chat_history: 会話履歴

    Returns:
        tuple: (回答文, 検索結果リスト[(Document, score), ...])
    """
    if st.session_state.rag_graph is None:
        st.session_state.rag_graph = build_rag_graph()

    initial_state = {
        "question": question,
        "k": k,
        "prompt_type": prompt_type,
        "chat_history": chat_history or [],
        "context_text": "",
        "retrieved_results": [],
        "answer": "",
        "search_query": question,
    }

    result = st.session_state.rag_graph.invoke(initial_state)

    return result["answer"], result["retrieved_results"]


# ============================================
# 補助関数：共通UI処理
# ============================================
def reset_chat_history():
    """会話履歴とLangGraphをリセット"""
    st.session_state.chat_history = []
    st.session_state.rag_graph = None


def get_history_for_prompt():
    """設定に応じて会話履歴を返す"""
    return st.session_state.chat_history if st.session_state.use_chat_history else []


def update_answer_state(answer, retrieved_results):
    """回答結果をSession Stateへ反映"""
    st.session_state.update(
        {
            "last_answer": answer,
            "last_retrieved_results": retrieved_results,
            "last_retrieved_docs": [doc for doc, score in retrieved_results],
        }
    )


def append_chat_history(question, answer):
    """今回の質問と回答を履歴へ保存"""
    if st.session_state.use_chat_history:
        st.session_state.chat_history.append({"question": question, "answer": answer})


def render_chunk_preview(chunks):
    """分割済みチャンクのプレビューを表示"""
    if not chunks:
        return

    st.subheader("📋 分割結果")
    st.info(
        f"分割方式: **{st.session_state.splitter_type}** / "
        f"総チャンク数: **{len(chunks)}**件"
    )

    preview_n = st.slider("プレビュー", 1, min(10, len(chunks)), 3)

    for i, doc in enumerate(chunks[:preview_n]):
        with st.expander(f"チャンク #{i+1}"):
            st.text(
                doc.page_content[:1000] + "..."
                if len(doc.page_content) > 1000
                else doc.page_content
            )
            st.caption(f"📄 {doc.metadata.get('source', '不明')}")


def render_chat_history_view():
    """会話履歴を表示"""
    if not st.session_state.chat_history:
        return

    st.subheader("🗂️ 会話履歴")
    for i, turn in enumerate(reversed(st.session_state.chat_history), start=1):
        history_index = len(st.session_state.chat_history) - i + 1
        with st.expander(f"会話履歴 #{history_index}"):
            st.markdown(f"**質問**: {turn['question']}")
            st.markdown(f"**回答**: {turn['answer']}")


def render_retrieved_results_view():
    """検索根拠を表示"""
    if not st.session_state.last_retrieved_results:
        return

    actual_k = len(st.session_state.last_retrieved_results)
    selected_k = st.session_state.retrieval_k
    st.subheader(f"🔍 検索根拠（取得 {actual_k}件 / 設定 k={selected_k}）")

    for i, (doc, score) in enumerate(st.session_state.last_retrieved_results):
        with st.expander(f"根拠 #{i+1}"):
            st.text(doc.page_content)
            st.caption(f"📄 {doc.metadata.get('source', '不明')}")
            st.caption(f"📏 類似度スコア（distance）: {score:.4f}")
            st.caption("※ Chromaのscoreは距離のため、小さいほど関連性が高い")


def render_splitter_comparison(uploaded_file, chunk_size, chunk_overlap):
    """分割方式・条件の比較表示"""
    show_splitter_comparison = st.checkbox(
        "分割方式の比較を表示する",
        value=st.session_state.show_splitter_comparison,
        help="オンにすると、現在設定と比較設定のチャンク分割結果を表示します。",
    )

    st.session_state.show_splitter_comparison = show_splitter_comparison
    if not uploaded_file or not st.session_state.show_splitter_comparison:
        return

    st.subheader("🧪 分割方式の比較")
    st.caption(
        "分割方式やパラメータの違いで、チャンク数や切れ方がどう変わるかを確認できます。"
    )

    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        compare_splitter_type = st.selectbox(
            "比較用の分割方式",
            SPLITTER_TYPES,
            key="compare_splitter_type",
            help="現在の設定と比較したい分割方式を選びます。",
        )

    with compare_col2:
        compare_chunk_size = st.slider(
            "比較用チャンクサイズ",
            300,
            1500,
            st.session_state.compare_chunk_size,
            100,
            key="compare_chunk_size_slider",
        )

    compare_chunk_overlap = st.slider(
        "比較用重なり幅",
        0,
        300,
        st.session_state.compare_chunk_overlap,
        20,
        key="compare_chunk_overlap_slider",
    )

    st.session_state.compare_chunk_size = compare_chunk_size
    st.session_state.compare_chunk_overlap = compare_chunk_overlap

    try:
        docs_for_compare = read_uploaded_file(uploaded_file)

        current_split_docs = split_documents(
            docs_for_compare, st.session_state.splitter_type, chunk_size, chunk_overlap
        )

        compare_split_docs = split_documents(
            docs_for_compare,
            compare_splitter_type,
            compare_chunk_size,
            compare_chunk_overlap,
        )

        st.markdown("### 比較結果")
        st.table(
            [
                {
                    "条件": "現在の設定",
                    "分割方式": st.session_state.splitter_type,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "チャンク数": len(current_split_docs),
                },
                {
                    "条件": "比較設定",
                    "分割方式": compare_splitter_type,
                    "chunk_size": compare_chunk_size,
                    "chunk_overlap": compare_chunk_overlap,
                    "チャンク数": len(compare_split_docs),
                },
            ]
        )

        tab1, tab2 = st.tabs(["現在の先頭5チャンク", "比較設定の先頭5チャンク"])

        with tab1:
            if current_split_docs:
                preview_count_current = min(5, len(current_split_docs))
                for i, doc in enumerate(current_split_docs[:preview_count_current]):
                    with st.expander(f"現在の設定 チャンク #{i+1}"):
                        st.text(
                            doc.page_content[:1000] + "..."
                            if len(doc.page_content) > 1000
                            else doc.page_content
                        )

        with tab2:
            if compare_split_docs:
                preview_count_compare = min(5, len(compare_split_docs))
                for i, doc in enumerate(compare_split_docs[:preview_count_compare]):
                    with st.expander(f"比較設定 チャンク #{i+1}"):
                        st.text(
                            doc.page_content[:1000] + "..."
                            if len(doc.page_content) > 1000
                            else doc.page_content
                        )

    except Exception as e:
        st.warning(f"比較表示でエラーが発生しました: {str(e)}")


def handle_index_creation(uploaded_file, chunk_size, chunk_overlap):
    """インデックス作成処理"""
    if not uploaded_file:
        st.warning("❌ ファイルをアップロードしてください")
        return
    elif not check_api_key():
        st.error("❌ .env に OPENAI_API_KEY を設定してください")
        return

    try:
        with st.spinner("文書処理中...（初回は時間がかかります）"):
            # 1. ファイル読み込み
            docs = read_uploaded_file(uploaded_file)

            # 2. 分割＆ベクトルDB作成
            _, split_docs = build_vectorstore(
                docs, chunk_size, chunk_overlap, st.session_state.splitter_type
            )

            # 3. 状態更新
            st.session_state.update(
                {
                    "vectorstore_ready": True,
                    "chunks": split_docs,
                    "last_answer": "",
                    "last_retrieved_docs": [],
                    "last_retrieved_results": [],
                }
            )

        st.success(f"✅ 完了！チャンク数: **{len(split_docs)}**件")
        st.balloons()  # 成功エフェクト

    except Exception as e:
        st.error(f"❌ エラー: {str(e)}")
        st.caption("ファイル形式やAPIキーを確認してください")


def handle_answer_generation(question):
    """回答生成処理"""
    if not st.session_state.vectorstore_ready:
        st.warning("⚠️ まず「インデックス作成」を実行してください")
        return
    elif not question.strip():
        st.warning("⚠️ 質問を入力してください")
        return
    elif not check_api_key():
        st.error("❌ .env の OPENAI_API_KEY を確認")
        return

    try:
        with st.spinner("検索中...回答生成中..."):
            # 設定に応じて履歴を使い分ける
            history_for_prompt = get_history_for_prompt()

            answer, retrieved_results = answer_question(
                question,
                st.session_state.retrieval_k,
                st.session_state.prompt_type,
                history_for_prompt,
            )

            # 状態更新
            update_answer_state(answer, retrieved_results)

            # 今回の質問と回答を履歴へ保存
            append_chat_history(question, answer)

        st.success("✅ 回答完了！")

    except Exception as e:
        st.error(f"❌ 回答生成エラー: {str(e)}")


# ============================================
# メインUI（左右2カラム）
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
        help="学習用サンプル: data/sample_company_handbook.txt",
    )

    splitter_type = st.selectbox(
        "✂️ 分割方式",
        SPLITTER_TYPES,
        index=SPLITTER_TYPES.index(st.session_state.splitter_type),
        help="分割方式を変えると、チャンクの切れ方や検索品質が変わることがあります。",
    )

    st.session_state.splitter_type = splitter_type

    # 分割設定
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "📏 チャンクサイズ",
            300,
            1500,
            700,
            100,
            help="1チャンクの文字数（大きい=文脈多め、小さい=細かい検索）",
        )

    with col2:
        chunk_overlap = st.slider(
            "🔗 重なり幅", 0, 300, 100, 20, help="チャンク間の重なり（文脈切れを防ぐ）"
        )

    # インデックス作成ボタン
    if st.button("🚀 インデックス作成", use_container_width=True):
        handle_index_creation(uploaded_file, chunk_size, chunk_overlap)

    # チャンク確認エリア
    render_chunk_preview(st.session_state.chunks)
    render_splitter_comparison(uploaded_file, chunk_size, chunk_overlap)

# ============================================
# 右カラム：質問＆回答（Retrieve→Generate）
# ============================================
with right_col:
    st.header("💬 質問する")
    st.caption(f"現在の検索件数 k = {st.session_state.retrieval_k}")

    # ユーザーが k を1〜10の範囲で調整可能
    retrieval_k = st.slider(
        "🔎 検索件数 k",
        min_value=1,
        max_value=10,
        value=st.session_state.retrieval_k,
        step=1,
        help="質問に対して、関連チャンクを何件取得するかを指定します。",
    )

    prompt_type = st.selectbox(
        "🧠 プロンプトタイプ",
        PROMPT_TYPES,
        index=PROMPT_TYPES.index(st.session_state.prompt_type),
        help="同じ検索結果でも、プロンプトにより回答スタイルが変わります。",
    )

    st.session_state.prompt_type = prompt_type
    st.session_state.retrieval_k = retrieval_k

    # 単発RAGと対話型RAGを切り替えるUI
    use_chat_history = st.checkbox(
        "🧠 会話履歴を使う",
        value=st.session_state.use_chat_history,
        help="オンにすると、前の質問と回答を参考にして次の質問へつなげます。",
    )

    st.session_state.use_chat_history = use_chat_history

    # 履歴を手動でリセット
    if st.button("🗑️ 会話履歴をクリア", use_container_width=True):
        reset_chat_history()
        st.success("会話履歴をクリアしました")

    # 質問入力
    question = st.text_input(
        "❓ 質問を入力",
        placeholder="例: 「リモートワークは何日まで可能？」",
        help="アップロードした文書に関する質問をどうぞ",
    )

    # 回答生成ボタン
    if st.button("🤖 回答生成", use_container_width=True):
        handle_answer_generation(question)

    # 回答表示
    if st.session_state.last_answer:
        st.subheader("📝 最終回答")
        st.caption(f"使用プロンプト: {st.session_state.prompt_type}")
        st.markdown(f"**{st.session_state.last_answer}**")

    # 現在の会話履歴を表示
    render_chat_history_view()

    # 検索根拠表示
    render_retrieved_results_view()

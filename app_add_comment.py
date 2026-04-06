# ============================================
# 1. ライブラリの読み込み
# ============================================
# このセクションでは、アプリで使うPythonライブラリを読み込みます。
# Streamlit は画面UI用、LangChain はRAGの部品用、dotenv は .env 読み込み用です。

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate


# ============================================
# 2. 環境変数の読み込み
# ============================================
# .env に書いた OPENAI_API_KEY を使えるようにします。
# APIキーをコードに直接書かずに済むため、安全で管理しやすくなります。

load_dotenv()


# ============================================
# 3. アプリ全体の基本設定
# ============================================
# ここではタイトルや保存先フォルダなど、
# アプリ全体で使う設定値をまとめて定義します。

APP_TITLE = "📚 LangChainで学ぶRAGチュートリアル"
PERSIST_DIR = Path("chroma_db")

st.set_page_config(
    page_title="LangChain RAG Tutorial",
    page_icon="📚",
    layout="wide"
)

st.title(APP_TITLE)
st.caption("初心者向けに、RAGの流れを見える化したStreamlitアプリ")


# ============================================
# 4. Session State の初期化
# ============================================
# Streamlit はボタン操作のたびにスクリプトを最初から再実行します。
# そのままだと前回の結果が消えるので、session_state に値を保存します。
# これにより、チャンク一覧や最終回答を画面に残せます。

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# ============================================
# 5. 補助関数：APIキー確認
# ============================================
# OpenAI APIを使う前に、APIキーが設定されているかを確認します。
# これを事前にチェックすると、原因のわかりやすいエラーメッセージを出せます。

def check_api_key():
    """
    OPENAI_API_KEY が設定されているかを確認する。

    Returns:
        bool: 設定されていれば True、未設定なら False
    """
    return bool(os.getenv("OPENAI_API_KEY"))


# ============================================
# 6. 補助関数：アップロードファイルを読む
# ============================================
# Streamlit でアップロードされたファイルを読み込み、
# LangChain が扱いやすい Document 形式に変換します。
# metadata にファイル名を残しておくと、後で出典表示に使えます。

def read_uploaded_file(uploaded_file):
    """
    アップロードされた txt / md ファイルを読み込み、
    LangChain の Document オブジェクトに変換する。

    Args:
        uploaded_file: st.file_uploader() で受け取ったファイル

    Returns:
        list[Document]: Document のリスト

    Raises:
        ValueError: 対応していない拡張子の場合
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in [".txt", ".md"]:
        text = uploaded_file.getvalue().decode("utf-8")
        return [
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name}
            )
        ]

    raise ValueError("対応しているファイル形式は .txt と .md のみです。")


# ============================================
# 7. 補助関数：文書を分割してベクトルDBを作る
# ============================================
# ここがRAGの前処理です。
# 長い文書をチャンクに分割し、各チャンクを埋め込みベクトルに変換して、
# Chroma ベクトルDB に保存します。
#
# LangChainのRAGでも、
# 「Load → Split → Store」の流れが基本になります。

def build_vectorstore(documents, chunk_size, chunk_overlap):
    """
    文書をチャンク分割し、埋め込みを作り、Chroma に保存する。

    Args:
        documents (list[Document]): 読み込んだ文書
        chunk_size (int): 1チャンクの文字数
        chunk_overlap (int): 前後チャンクの重なり文字数

    Returns:
        tuple:
            vectorstore: 作成した Chroma ベクトルDB
            split_docs: 分割後の Document リスト
    """
    # 長い文書を小さなチャンクに分割します。
    # chunk_overlap を少し入れると、文脈の切れ目が不自然になりにくいです。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)

    # 学習用アプリでは毎回まっさらな状態で確認したいので、
    # 以前のベクトルDBがあれば削除して作り直します。
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    # 各チャンクをベクトル化する埋め込みモデルを作成します。
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 分割したチャンクをChromaに保存します。
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR)
    )

    return vectorstore, split_docs


# ============================================
# 8. 補助関数：質問に回答する
# ============================================
# ここがRAGの本番処理です。
# ベクトルDBから質問に近いチャンクを検索し、
# その検索結果だけを文脈としてLLMに渡して回答を作ります。
#
# この流れは、
# 「Retrieve → Generate」に対応しています。

def answer_question(question):
    """
    質問に対してRAGで回答を生成する。

    Args:
        question (str): ユーザーの質問

    Returns:
        tuple:
            response.content (str): 生成された回答
            retrieved_docs (list[Document]): 検索された関連チャンク
    """
    # 保存済みのベクトルDBを読み込みます。
    # 検索時にも同じ埋め込みモデルを使います。
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # Retriever は「質問に近い文書を探す役割」です。
    # k=3 なので、上位3件の関連チャンクを取得します。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)

    # 検索されたチャンクをつなげて、LLMに渡す参考文脈を作ります。
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # LLMへの指示文です。
    # 「参考文脈だけを使う」「答えがなければわからないと言う」
    # というルールを与えています。
    prompt = ChatPromptTemplate.from_template(
        """
あなたは初心者にもわかりやすく説明する親切なAIアシスタントです。
以下の参考文脈だけを使って質問に答えてください。
文脈に答えがない場合は、わからないと明確に伝えてください。
専門用語が出る場合は、なるべくやさしく説明してください。

参考文脈:
{context}

質問:
{question}
"""
    )

    # 回答生成に使うチャットモデルです。
    # temperature=0 にすると、回答のブレを小さくしやすいです。
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    # LangChain のチェーンを作ります。
    chain = prompt | llm

    # 実際に回答を生成します。
    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    return response.content, retrieved_docs


# ============================================
# 9. 説明エリア
# ============================================
# ここでは、RAG と LangChain の基本概念を
# 初学者向けに文章で説明しています。

with st.expander("RAGとLangChainをやさしく理解する", expanded=True):
    st.markdown("""
### RAGとは？
RAG（Retrieval-Augmented Generation）は、AIが答える前に、関連する文書を検索してから回答する仕組みです。

### なぜ便利？
LLMは便利ですが、あなたの手元の資料やローカルファイルを最初から知っているわけではありません。
そこで、質問に合う文書を検索して、その内容を参考に答えます。

### LangChainとは？
LangChainは、LLMアプリを作るための部品をまとめたフレームワークです。
たとえば次のような処理をつなぎやすくしてくれます。

1. 文書を読み込む
2. 長い文書を小さく分割する
3. 検索しやすい形に変換する
4. 関連する文書を検索する
5. その結果をもとに回答する

### このアプリで体験できること
- テキスト文書をアップロードする
- 文書をチャンクに分割する
- ベクトルDBに保存する
- 質問に近いチャンクを検索する
- 検索結果を使って回答を作る
""")

st.markdown("---")


# ============================================
# 10. 画面レイアウト
# ============================================
# 左右2カラムに分けます。
# 左側は「文書準備」、右側は「質問と回答」です。

left, right = st.columns([1, 1])


# ============================================
# 11. 左カラム：文書アップロードと前処理
# ============================================
with left:
    st.subheader("① 文書を準備する")

    # 学習用として、シンプルに txt / md のみ受け付けます。
    uploaded_file = st.file_uploader(
        "txt または md ファイルをアップロードしてください",
        type=["txt", "md"],
        help="学習用として、まずはシンプルなテキストファイルだけに絞っています。"
    )

    # chunk_size は1つのチャンクの大きさです。
    # 大きいと1回に多くの文脈を持てますが、
    # 検索が粗くなることもあります。
    chunk_size = st.slider(
        "chunk_size（1チャンクの文字数）",
        min_value=300,
        max_value=1500,
        value=700,
        step=100
    )

    # chunk_overlap は前後チャンクの重なり量です。
    # 少し重ねると文の途中で意味が切れにくくなります。
    chunk_overlap = st.slider(
        "chunk_overlap（チャンクの重なり）",
        min_value=0,
        max_value=300,
        value=100,
        step=20
    )

    # ボタンを押すと前処理を実行します。
    if st.button("インデックスを作成", use_container_width=True):
        if uploaded_file is None:
            st.warning("先にファイルをアップロードしてください。")
        elif not check_api_key():
            st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
        else:
            try:
                with st.spinner("文書を読み込み、分割し、ベクトルDBを作成しています..."):
                    # 1. ファイルを読む
                    documents = read_uploaded_file(uploaded_file)

                    # 2. 分割してベクトルDBを作る
                    _, split_docs = build_vectorstore(
                        documents,
                        chunk_size,
                        chunk_overlap
                    )

                    # 3. 状態を保存する
                    st.session_state.vectorstore_ready = True
                    st.session_state.chunks = split_docs

                    # 4. 新しい文書で作り直したので、過去の結果はリセットする
                    st.session_state.last_retrieved_docs = []
                    st.session_state.last_answer = ""

                st.success(f"インデックス作成が完了しました。チャンク数: {len(split_docs)}")
            except Exception as e:
                st.error(f"インデックス作成中にエラーが発生しました: {e}")

    # 分割結果の確認エリアです。
    # 学習用アプリでは、ここを見てチャンクの切れ方を確認するのが大切です。
    if st.session_state.chunks:
        st.subheader("② 分割されたチャンクを確認")

        st.write(f"チャンク数: {len(st.session_state.chunks)}")

        preview_count = st.slider(
            "表示するチャンク数",
            min_value=1,
            max_value=min(10, len(st.session_state.chunks)),
            value=min(3, len(st.session_state.chunks))
        )

        for i, doc in enumerate(st.session_state.chunks[:preview_count]):
            with st.expander(f"チャンク {i + 1}"):
                st.write(doc.page_content)
                st.caption(f"source: {doc.metadata.get('source', 'unknown')}")


# ============================================
# 12. 右カラム：質問入力と回答生成
# ============================================
with right:
    st.subheader("③ 質問してみる")

    # 文書に対する質問を入力してもらいます。
    question = st.text_input(
        "文書について質問してください",
        placeholder="例: この文書の要点を初心者向けに説明してください"
    )

    # ボタンを押すと、検索して回答を作ります。
    if st.button("回答を生成", use_container_width=True):
        if not st.session_state.vectorstore_ready:
            st.warning("先にインデックスを作成してください。")
        elif not question.strip():
            st.warning("質問を入力してください。")
        elif not check_api_key():
            st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
        else:
            try:
                with st.spinner("関連チャンクを検索して回答を生成しています..."):
                    # RAG実行
                    answer, retrieved_docs = answer_question(question)

                    # 結果を保存
                    st.session_state.last_answer = answer
                    st.session_state.last_retrieved_docs = retrieved_docs
            except Exception as e:
                st.error(f"回答生成中にエラーが発生しました: {e}")

    # 最終回答の表示
    if st.session_state.last_answer:
        st.subheader("④ 最終回答")
        st.write(st.session_state.last_answer)

    # 検索された関連チャンクの表示
    # 「どの文書を根拠に答えたか」を見せることで、
    # RAGの流れを理解しやすくします。
    if st.session_state.last_retrieved_docs:
        st.subheader("検索された関連チャンク")

        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            with st.expander(f"検索結果 {i + 1}"):
                st.write(doc.page_content)
                st.caption(f"source: {doc.metadata.get('source', 'unknown')}")


# ============================================
# 13. 学習ポイントの補足
# ============================================
# 最後に、RAG学習で重要な調整ポイントを簡単に表示します。

st.markdown("---")
st.markdown("""
### 学習ポイント
- chunk_size を大きくすると、1回に扱う文脈が増えます
- chunk_overlap を入れると、文脈の切れ目を少しなめらかにできます
- RAGでは「最終回答」だけでなく「何が検索されたか」を確認することが大切です
""")

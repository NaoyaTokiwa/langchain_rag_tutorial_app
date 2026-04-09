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

load_dotenv()

APP_TITLE = "📚 LangChainで学ぶRAGチュートリアル"
PERSIST_DIR = Path("chroma_db")

st.set_page_config(
    page_title="LangChain RAG Tutorial",
    page_icon="📚",
    layout="wide"
)

st.title(APP_TITLE)
st.caption("初心者向けに、RAGの流れを見える化したStreamlitアプリ")

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

def check_api_key():
    return bool(os.getenv("OPENAI_API_KEY"))

def read_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in [".txt", ".md"]:
        text = uploaded_file.getvalue().decode("utf-8")
        return [Document(page_content=text, metadata={"source": uploaded_file.name})]

    raise ValueError("対応しているファイル形式は .txt と .md のみです。")

def build_vectorstore(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)

    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR)
    )

    return vectorstore, split_docs

def answer_question(question):
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

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

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    chain = prompt | llm
    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    return response.content, retrieved_docs

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

left, right = st.columns([1, 1])

with left:
    st.subheader("① 文書を準備する")

    uploaded_file = st.file_uploader(
        "txt または md ファイルをアップロードしてください",
        type=["txt", "md"],
        help="学習用として、まずはシンプルなテキストファイルだけに絞っています。"
    )

    chunk_size = st.slider(
        "chunk_size（1チャンクの文字数）",
        min_value=300,
        max_value=1500,
        value=700,
        step=100
    )

    chunk_overlap = st.slider(
        "chunk_overlap（チャンクの重なり）",
        min_value=0,
        max_value=300,
        value=100,
        step=20
    )

    if st.button("インデックスを作成", use_container_width=True):
        if uploaded_file is None:
            st.warning("先にファイルをアップロードしてください。")
        elif not check_api_key():
            st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
        else:
            try:
                with st.spinner("文書を読み込み、分割し、ベクトルDBを作成しています..."):
                    documents = read_uploaded_file(uploaded_file)
                    _, split_docs = build_vectorstore(documents, chunk_size, chunk_overlap)

                    st.session_state.vectorstore_ready = True
                    st.session_state.chunks = split_docs
                    st.session_state.last_retrieved_docs = []
                    st.session_state.last_answer = ""

                st.success(f"インデックス作成が完了しました。チャンク数: {len(split_docs)}")
            except Exception as e:
                st.error(f"インデックス作成中にエラーが発生しました: {e}")

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

with right:
    st.subheader("③ 質問してみる")

    question = st.text_input(
        "文書について質問してください",
        placeholder="例: この文書の要点を初心者向けに説明してください"
    )

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
                    answer, retrieved_docs = answer_question(question)
                    st.session_state.last_answer = answer
                    st.session_state.last_retrieved_docs = retrieved_docs
            except Exception as e:
                st.error(f"回答生成中にエラーが発生しました: {e}")

    if st.session_state.last_answer:
        st.subheader("④ 最終回答")
        st.write(st.session_state.last_answer)

    if st.session_state.last_retrieved_docs:
        st.subheader("検索された関連チャンク")

        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            with st.expander(f"検索結果 {i + 1}"):
                st.write(doc.page_content)
                st.caption(f"source: {doc.metadata.get('source', 'unknown')}")

st.markdown("---")
st.markdown("""
### 学習ポイント
- chunk_size を大きくすると、1回に扱う文脈が増えます
- chunk_overlap を入れると、文脈の切れ目を少しなめらかにできます
- RAGでは「最終回答」だけでなく「何が検索されたか」を確認することが大切です
""")

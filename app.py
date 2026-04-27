# ============================================
# ライブラリの読み込み
# ============================================
# Streamlit: 画面UI作成用
# LangChain: RAG処理の部品群
# python-dotenv: .envファイル読み込み用

import os
import shutil  # ファイル操作
import uuid  # 一意ID生成
from pathlib import Path  # パス操作
from typing import TypedDict  # 型付き辞書

import streamlit as st  # WebアプリUI
from dotenv import load_dotenv  # 環境変数読み込み
from langchain.memory import ConversationSummaryMemory  # 会話履歴の要約メモリ
from langchain.prompts import ChatPromptTemplate, PromptTemplate  # プロンプト作成
from langchain_community.vectorstores import Chroma  # ベクトルDB

# LangChain系（RAGの部品箱）
from langchain_core.documents import Document  # 文書データ型
from langchain_core.messages import SystemMessage  # メッセージ型
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser  # 文字列変換
from langchain_core.tools import tool  # Tool化デコレータ
from langchain_openai import ChatOpenAI  # OpenAIチャットモデル
from langchain_openai import OpenAIEmbeddings  # 埋め込みモデル
from langchain_text_splitters import CharacterTextSplitter  # 単純分割
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 再帰分割

# LangGraph系（状態付き対話フロー）
from langgraph.graph import END, START, StateGraph  # グラフ制御

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
PERSIST_ROOT_DIR = Path("chroma_db")  # ベクトルDB保存フォルダ
PROMPT_TYPES = ["初心者向け", "要約重視", "箇条書き重視"]
SPLITTER_TYPES = ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"]
MODE_TYPES = [
    "通常RAG",
    "Function Calling RAG",
    "LLMルーティング RAG",
]  # UI上で通常RAG / Function Calling RAG / Query Routing RAG を切り替え
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
    "conversation_memory": None,  # ConversationSummaryMemory本体
    "use_chat_history": True,
    "memory_summary_enabled": True,  # 要約メモリを使うかどうか
    "memory_max_turns_before_summary": 3,  # 要約に加えて生で保持する直近ターン数
    "rag_graph": None,
    "persist_dir": None,
    "tool_calling_graph": None,  # Function Calling 用のLangGraph保持用
    "execution_mode": "通常RAG",  # UI の選択状態
    "last_tool_trace": [],  # ログ表示用
    "workflow_routing_graph": None,  # LLM Routing 用のLangGraph保持用
    "last_workflow_route": "",  # LLM Routingで選ばれた経路表示用
    "last_web_context": "",  # Web検索ルートの参考メモ表示用
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
    """Session Stateに未設定の初期値を投入する。

    Args:
        なし。

    Returns:
        None: `st.session_state` に `SESSION_DEFAULTS` の不足キーを追加する。
    """
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================
# 補助関数：APIキー確認
# ============================================
def check_api_key():
    """`OPENAI_API_KEY` の設定有無を判定する。

    Args:
        なし。

    Returns:
        bool: 環境変数 `OPENAI_API_KEY` が設定されていれば `True`、未設定または空文字なら `False`。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key)  # 空文字でないことを確認


# ============================================
# 補助関数：ファイル読み込み（Load）
# ============================================
# RAGの最初のステップ「文書読み込み」です。
# Streamlitのfile_uploaderから受け取ったファイルを
# LangChainのDocument形式に変換します。


def read_uploaded_file(uploaded_file):
    """アップロード済みテキストファイルを `Document` のリストへ変換する。

    Args:
        uploaded_file: `st.file_uploader()` が返すアップロードファイルオブジェクト。

    Returns:
    list[Document]: `page_content` に本文、`metadata["source"]` に
        ファイル名を持つ `Document` を1件だけ含むリスト。

    Raises:
        ValueError: 拡張子が `.txt` または `.md` 以外の場合。
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
    """指定した分割方式で文書をチャンク分割する。

    Args:
        documents: 分割対象の `list[Document]`。
        splitter_type: 使用する分割器名。
        `"RecursiveCharacterTextSplitter"` または `"CharacterTextSplitter"`。
        chunk_size: 1チャンクあたりの最大文字数。
        chunk_overlap: 前後チャンクで重複させる文字数。

    Returns:
        list[Document]: 分割後の `Document` リスト。

    Raises:
        ValueError: 未対応の `splitter_type` が指定された場合。
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
# 補助関数：永続化先作成
# ============================================
def create_persist_dir():
    """Chroma 用の永続化ディレクトリを新規作成する。

    Args:
        なし。

    Returns:
        Path: `chroma_db/index_<uuid>` 形式で作成された保存先ディレクトリ。
    """
    PERSIST_ROOT_DIR.mkdir(exist_ok=True)
    persist_dir = PERSIST_ROOT_DIR / f"index_{uuid.uuid4().hex}"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


# ============================================
# 補助関数：古いインデックスのクリーンアップ
# ============================================
def cleanup_old_persist_dirs(current_persist_dir=None):
    """現在使用中以外の古い Chroma 保存先ディレクトリを削除する。

    Args:
        current_persist_dir: 削除対象から除外する `Path`。`None` の場合は全ディレクトリが削除候補。

    Returns:
        None: 不要な永続化ディレクトリを削除する。
    """
    if not PERSIST_ROOT_DIR.exists():
        return

    for path in PERSIST_ROOT_DIR.iterdir():
        if not path.is_dir():
            continue
        if current_persist_dir is not None and path == current_persist_dir:
            continue
        shutil.rmtree(path, ignore_errors=True)


# ============================================
# 補助関数：ベクトルDB作成（Split + Store）
# ============================================
# RAGの前処理「文書分割→埋め込み→保存」です。
# LangChainの基本フロー「Load→Split→Store」に相当。


def build_vectorstore(
    documents, chunk_size, chunk_overlap, splitter_type="RecursiveCharacterTextSplitter"
):
    """文書の分割・埋め込み・Chroma 保存をまとめて実行する。

    Args:
        documents: ベクトル化対象の `list[Document]`。
        chunk_size: 1チャンクあたりの最大文字数。
        chunk_overlap: 前後チャンクで重複させる文字数。
        splitter_type: 使用する分割器名。

    Returns:
        tuple[Chroma, list[Document], Path]:
            作成した `Chroma` ベクトルストア、分割後 `Document` リスト、永続化先ディレクトリの組。
    """
    # Step1: 文書分割（長い文章を小さく分割）
    split_docs = split_documents(documents, splitter_type, chunk_size, chunk_overlap)

    # Step2: 新しい永続化先を作成（既存DBは削除せず切り替える）
    persist_dir = create_persist_dir()

    # Step3: 埋め込みモデル作成（文章→数値ベクトル変換）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step4: ChromaベクトルDB作成＆保存
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),  # ディスク保存
    )

    return vectorstore, split_docs, persist_dir


# ============================================
# 補助関数：プロンプト取得
# ============================================
def get_prompt_template(prompt_type):
    """回答スタイルに応じた `ChatPromptTemplate` を返す。

    Args:
        prompt_type: `PROMPT_TYPES` のいずれかの文字列。

    Returns:
        ChatPromptTemplate: 回答生成用テンプレート。
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
    """直近会話履歴を LLM に渡しやすい文字列へ整形する。

    Args:
        chat_history: `[{"question": str, "answer": str}, ...]` 形式の会話履歴。
        max_turns: 末尾から何ターン分を含めるか。

    Returns:
        str: 整形済み会話履歴。履歴が空なら `"会話履歴なし"`。
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


SUMMARY_PROMPT_JA = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
あなたは会話履歴を日本語で要約するアシスタントです。
これまでの要約と新しい会話履歴をもとに、要点だけを自然な日本語で更新要約してください。
必ず日本語で出力してください。
箇条書きではなく、読みやすい短い文章でまとめてください。

これまでの要約:
{summary}

新しい会話履歴:
{new_lines}

更新後の要約:
""".strip(),
)


# ============================================
# 補助関数：ConversationSummaryMemory関連
# ============================================
# 長い会話履歴をそのまま全部渡すと、トークン数が増えやすくなります。
# そこで LangChain の ConversationSummaryMemory を使い、
# 「これまでの会話要約」+「直近の数ターン」を LLM に渡す構成にします。
def get_or_create_conversation_memory():
    """`ConversationSummaryMemory` を取得し、未作成なら初期化する。

    Args:
        なし。

    Returns:
        ConversationSummaryMemory: Session State 上で共有される要約メモリ。
    """
    if st.session_state.conversation_memory is None:
        summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        st.session_state.conversation_memory = ConversationSummaryMemory(
            llm=summary_llm,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,
            prompt=SUMMARY_PROMPT_JA,  # ← 日本語要約プロンプトを指定
        )
    return st.session_state.conversation_memory


def rebuild_conversation_memory_from_history(chat_history):
    """現在の会話履歴から要約メモリを再構築する。

    Notes:
        Streamlit の再実行により `chat_history` と内部メモリがずれる可能性があるため、
        毎回履歴を正として `ConversationSummaryMemory` を作り直せるようにしている。

    Args:
        chat_history: `[{"question": str, "answer": str}, ...]` 形式の会話履歴。

    Returns:
        ConversationSummaryMemory: `chat_history` の内容で再構築された要約メモリ。
    """
    memory = get_or_create_conversation_memory()
    memory.clear()  # 古い要約や重複した会話が残らないように一旦クリアする

    for turn in chat_history:  # 今の履歴一覧から要約メモリを再生成
        memory.save_context(
            {"input": turn["question"]},
            {"output": turn["answer"]},
        )

    return memory


def format_chat_history_with_summary(chat_history, max_turns=3):
    """会話履歴を「要約 + 直近ターン」の文字列へ整形する。

    Args:
        chat_history: `[{"question": str, "answer": str}, ...]` 形式の会話履歴。
        max_turns: 要約とは別に生のまま残す直近ターン数。

    Returns:
        str: 要約付きの会話履歴テキスト。履歴が短い場合は直近履歴のみ返す。
    """
    if not chat_history:
        return "会話履歴なし"

    # 要約機能をオフにした場合は、従来どおり直近履歴だけを返します。
    if not st.session_state.memory_summary_enabled:
        return format_chat_history(chat_history, max_turns=max_turns)

    memory = rebuild_conversation_memory_from_history(chat_history)
    memory_variables = memory.load_memory_variables({})
    summary_text = memory_variables.get("chat_history", "")

    recent_history = chat_history[-max_turns:]  # 要約とは別に、直近ターンは生で保持する
    recent_history_texts = []
    for i, turn in enumerate(recent_history, start=1):
        recent_history_texts.append(
            f"[直近の会話 {i}]\n質問: {turn['question']}\n回答: {turn['answer']}"
        )

    # まだ会話が短い場合は、無理に要約を前面に出さず直近履歴のみ返します。
    if len(chat_history) <= max_turns:
        return "\n\n".join(recent_history_texts)

    return (
        "[これまでの会話要約]\n"
        f"{summary_text if summary_text else '要約なし'}\n\n"
        "[直近の会話]\n" + "\n\n".join(recent_history_texts)
    )


# ============================================
# LLMによる処理フロー分岐用の補助関数
# ============================================
# 質問内容に応じて、
# - document: アップロード文書を検索して答える
# - web: 外部情報をもとに答える
# - general: 検索せず通常応答する
# の3つの処理フローへ振り分ける。


def classify_workflow_route_with_llm(question, chat_history):
    """質問内容から処理フロー種別を LLM で分類する。

    Args:
        question: 現在のユーザー質問。
        chat_history: `list[dict]` 形式の会話履歴。

    Returns:
        str: `"document"`、`"web"`、`"general"` のいずれか。
    """
    # 分類ぶれを減らしたいので temperature=0 にする
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 曖昧な follow-up 質問も判定しやすいように履歴を渡す
    history_text = format_chat_history_with_summary(
        chat_history, max_turns=st.session_state.memory_max_turns_before_summary
    )

    # 3つの候補から1語だけ返すよう明示して、後続の条件分岐を安定させる
    prompt = ChatPromptTemplate.from_template(
        """
あなたは質問に応じて処理フローを判定するアシスタントです。
以下の質問を見て、次の3種類のうち最も適切な経路を1語だけで返してください。

選択肢:
- document: アップロード済み文書の内容を検索すべき質問
- web: 最新情報、外部情報、一般Web情報を検索すべき質問
- general: 検索なしで通常のLLM応答で十分な質問

判定ルール:
- 社内規定、社内文書、アップロード資料、マニュアル、ハンドブックの内容は document
- 就業規則、人事制度、勤怠、休暇、給与、福利厚生、申請方法、勤務形態、出社頻度、リモートワーク可否など、会社や組織ごとのルールに依存する質問は document
- 文書にあるか不明でも、「この資料では」「この文書では」「アップロードした内容では」などが含まれるなら document
- ユーザー質問が短くても、社内制度や社内ルールの確認と解釈できる場合は document を優先
- 最新ニュース、現在の出来事、外部サービス情報、一般知識の確認は web
- 文章の言い換え、アイデア出し、概念説明、感想生成、相談は general
- 時事性がありそうなら web を優先

会話履歴:
{chat_history}

質問:
{question}

出力は document / web / general のいずれか1語のみ。
"""
    )

    chain = prompt | llm | StrOutputParser()
    route = (
        chain.invoke({"chat_history": history_text, "question": question})
        .strip()
        .lower()
    )

    if route not in ["document", "web", "general"]:
        return "document"

    return route


def search_web_context(question):
    """外部情報の調査メモ風コンテキストを生成する。

    Args:
        question: 現在のユーザー質問。

    Returns:
        str: 後段の回答生成で使う Web 調査メモ文字列。
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 学習用として、外部情報の「調査メモ」を先に作る
    # 実運用ではここを検索API呼び出しに置き換えやすい
    prompt = ChatPromptTemplate.from_template(
        """
あなたはWeb検索結果要約アシスタントです。
ユーザーの質問に答えるために必要な一般的な外部情報を、簡潔な箇条書きの調査メモとして作成してください。

重要ルール:
- これは学習用の擬似Web検索コンテキストです
- 断定しすぎず、必要に応じて「最新性は別途確認が必要」と添えてください
- 回答文ではなく、後段の回答生成で使うための参考メモとして出力してください
- 3〜5項目程度に整理してください

質問:
{question}
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


def general_answer_node_response(question, prompt_type, chat_history):
    """検索を使わない通常応答を生成する。

    Args:
        question: 現在のユーザー質問。
        prompt_type: 回答スタイルを表す文字列。
        chat_history: `list[dict]` 形式の会話履歴。

    Returns:
        str: 生成された回答文。
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history_text = format_chat_history_with_summary(
        chat_history, max_turns=st.session_state.memory_max_turns_before_summary
    )

    # general ルートでは検索を使わず、そのまま回答スタイルに沿って返す
    prompt = ChatPromptTemplate.from_template(
        """
あなたは親切なAIアシスタントです。
次の回答スタイルに従って、日本語でわかりやすく答えてください。

回答スタイル: {prompt_type}
会話履歴:
{chat_history}

質問:
{question}
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "prompt_type": prompt_type,
            "chat_history": history_text,
            "question": question,
        }
    )


# LLMによる処理フロー分岐用の状態。
# route に分類結果を持たせ、条件分岐で各ルートへ流す。
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


# ============================================
# Tool Calling用の外部ツール定義
# ============================================
# Function Callingを学べるように、LLM が必要に応じて呼び出せる
# Python関数を LangChain tool として定義する。
# ここでは「文書検索」と「会話履歴要約」の2種類を用意し、
# LangGraph から LLM -> Tool -> LLM の流れを体験できるようにしている。
@tool
def search_documents_tool(query: str) -> str:
    """
    ベクトルDBを検索し、関連チャンクを根拠テキストとして返す

    Args:
        query: 検索に使う質問文

    Returns:
        str: 検索結果を連結した根拠テキスト
    """
    if st.session_state.persist_dir is None:
        return "ベクトルDBが未作成です。先にインデックスを作成してください。"

    vectorstore = Chroma(
        persist_directory=str(st.session_state.persist_dir),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retrieved_results = vectorstore.similarity_search_with_score(
        query, k=st.session_state.retrieval_k
    )

    st.session_state.last_retrieved_results = retrieved_results
    st.session_state.last_retrieved_docs = [doc for doc, _ in retrieved_results]

    if not retrieved_results:
        return "関連文書が見つかりませんでした。"

    lines = []
    for i, (doc, score) in enumerate(retrieved_results, start=1):
        source = doc.metadata.get("source", "不明")
        snippet = doc.page_content[:500]
        lines.append(f"[根拠 {i}] source={source} distance={score:.4f}\n{snippet}")
    return "\n\n".join(lines)


@tool
def summarize_history_tool() -> str:
    """
    直近の会話履歴を検索や回答生成向けに短く要約して返す

    Returns:
        str: 整形済みの会話履歴テキスト
    """
    chat_history = st.session_state.chat_history
    if not chat_history:
        return "会話履歴なし"
    return format_chat_history_with_summary(
        chat_history, max_turns=st.session_state.memory_max_turns_before_summary
    )


TOOLS = [search_documents_tool, summarize_history_tool]


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
    persist_dir: str


# Function Calling用の状態。
# messages を持たせることで、LLMの応答・ToolMessage・再実行の流れを
# LangGraph上で追いやすくしている。
class ToolCallingState(TypedDict):
    question: str
    prompt_type: str
    chat_history: list
    answer: str
    tool_trace: list
    messages: list  # 通常RAGでは不要だった messages を保持するのがポイント


# ============================================
# LangGraphノード：質問補完
# ============================================
def rewrite_query_node(state: RAGState):
    """会話履歴を踏まえて検索用クエリを補完する。

    Args:
        state: `RAGState`。`question` と `chat_history` を含む状態辞書。

    Returns:
        dict[str, str]: 更新後の `search_query` を持つ辞書。
    """
    if not state["chat_history"]:
        return {"search_query": state["question"]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history_text = format_chat_history_with_summary(
        state["chat_history"],
        max_turns=st.session_state.memory_max_turns_before_summary,
    )

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
    """ベクトルストアから関連文書チャンクを検索する。

    Args:
        state: `RAGState`。`persist_dir`、`search_query`、`k` を含む状態辞書。

    Returns:
        dict[str, object]: `retrieved_results` と `context_text` を更新する辞書。
    """
    vectorstore = Chroma(
        persist_directory=state["persist_dir"],
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
    """検索結果と会話履歴から最終回答を生成する。

    Args:
        state: `RAGState`。`context_text`、`question`、`prompt_type` などを含む状態辞書。

    Returns:
        dict[str, str]: 更新後の `answer` を持つ辞書。
    """
    history_text = format_chat_history_with_summary(
        state["chat_history"],
        max_turns=st.session_state.memory_max_turns_before_summary,
    )
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
# LLMによる処理フロー分岐用ノード
# ============================================
# 最初に質問を分類し、その結果に応じて
# document / web / general の処理フローへ分岐する。


def classify_workflow_route_node(state: WorkflowRoutingState):
    """処理フロー分類結果を状態へ書き戻す。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, str]: `route` を持つ辞書。
    """
    route = classify_workflow_route_with_llm(state["question"], state["chat_history"])
    return {"route": route}


# document フローでは、既存RAGと同様に質問補完を行う。
# これにより、文書検索が必要な follow-up 質問にも対応しやすくなる。
def workflow_document_rewrite_query_node(state: WorkflowRoutingState):
    """document フロー用に検索クエリを補完する。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, str]: 更新後の `search_query` を持つ辞書。
    """
    if not state["chat_history"]:
        return {"search_query": state["question"]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history_text = format_chat_history_with_summary(
        state["chat_history"],
        max_turns=st.session_state.memory_max_turns_before_summary,
    )

    # 会話の流れを踏まえて、検索しやすい具体的な質問文へ補完する
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

    chain = prompt | llm | StrOutputParser()
    search_query = chain.invoke(
        {"chat_history": history_text, "question": state["question"]}
    )
    return {"search_query": search_query.strip()}


def workflow_document_retrieve_node(state: WorkflowRoutingState):
    """document フローで文書検索を実行する。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, object]: `retrieved_results` と `context_text` を持つ辞書。
    """
    vectorstore = Chroma(
        persist_directory=state["persist_dir"],
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    retrieved_results = vectorstore.similarity_search_with_score(
        state["search_query"], k=state["k"]
    )
    retrieved_docs = [doc for doc, _ in retrieved_results]
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    return {"retrieved_results": retrieved_results, "context_text": context_text}


def workflow_generate_document_answer_node(state: WorkflowRoutingState):
    """document フローで最終回答を生成する。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, str]: `answer` を持つ辞書。
    """
    history_text = format_chat_history_with_summary(
        state["chat_history"],
        max_turns=st.session_state.memory_max_turns_before_summary,
    )
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


def workflow_web_search_node(state: WorkflowRoutingState):
    """web フロー用の調査メモを生成する。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, object]: `web_context`、`context_text`、`retrieved_results` を持つ辞書。
    """
    web_context = search_web_context(state["question"])
    return {
        "web_context": web_context,
        "context_text": web_context,
        "retrieved_results": [],
    }


def workflow_generate_web_answer_node(state: WorkflowRoutingState):
    """web フローで調査メモをもとに回答を生成する。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, str]: `answer` を持つ辞書。
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history_text = format_chat_history_with_summary(
        state["chat_history"],
        max_turns=st.session_state.memory_max_turns_before_summary,
    )

    # web フローでは、事前に作った調査メモを根拠として回答を作る
    prompt = ChatPromptTemplate.from_template(
        """
あなたは親切なAIアシスタントです。
次のWeb調査メモを参考にして、日本語でわかりやすく回答してください。
必要に応じて、最新性は確認が必要である旨も短く添えてください。

回答スタイル: {prompt_type}
会話履歴:
{chat_history}

Web調査メモ:
{context}

質問:
{question}
"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "prompt_type": state["prompt_type"],
            "chat_history": history_text,
            "context": state["context_text"],
            "question": state["question"],
        }
    )
    return {"answer": answer}


def workflow_general_answer_node(state: WorkflowRoutingState):
    """general フローの通常応答を生成する。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        dict[str, object]: `answer`、`retrieved_results`、
            `context_text`、`web_context` を格納した辞書。
    """
    answer = general_answer_node_response(
        state["question"], state["prompt_type"], state["chat_history"]
    )
    return {
        "answer": answer,
        "retrieved_results": [],
        "context_text": "",
        "web_context": "",
    }


def decide_workflow_after_classification(state: WorkflowRoutingState):
    """分類結果に応じて次のノード名を返す。

    Args:
        state: `WorkflowRoutingState`。

    Returns:
        str: 次に遷移するノード名。
    """
    route = state.get("route", "document")

    if route == "web":
        return "workflow_web_search"
    if route == "general":
        return "workflow_general_answer"

    return "workflow_document_rewrite_query"


# ============================================
# Function Calling用ノード
# ============================================
# 1つ目のノードでは、LLM に「必要ならツールを呼んでよい」と指示し、
# bind_tools() によって Tool Calling を有効化する。
# これにより、LLM は search_documents_tool / summarize_history_tool を
# 自律的に選択して tool_calls を返せるようになる。
def tool_calling_llm_node(state: ToolCallingState):
    """Tool Calling 可能な LLM ノードを実行する。

    Args:
        state: `ToolCallingState`。`messages`、`question`、`chat_history` などを含む状態辞書。

    Returns:
        dict[str, list]: 更新後の `messages` を持つ辞書。
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
    history_text = format_chat_history_with_summary(
        state["chat_history"],
        max_turns=st.session_state.memory_max_turns_before_summary,
    )

    system_prompt = SystemMessage(
        content=f"""
        あなたはRAGアシスタントです。
        ユーザー質問には、必ずツールで取得した情報だけを根拠として回答してください。

        重要ルール:
        - 文書の根拠が必要なら search_documents_tool を使ってください
        - 会話履歴の要点整理が必要なら summarize_history_tool を使ってください
        - ツール結果に明示されていない内容を、一般知識や推測で補ってはいけません
        - 質問に答えるための根拠が不足している場合は、「わかりません」と答えてください
        - 特に制度、規定、金額、日数、連絡先、禁止事項は、必ず検索結果に基づいて答えてください
        - 検索すべき内容なのにまだ検索していない場合は、先に search_documents_tool を呼んでください

        回答スタイル: {state['prompt_type']}
        会話履歴:
        {history_text}

        最終回答では、ツール結果に含まれる情報だけを使って日本語で答えてください。
        """
    )

    messages = state.get("messages", [])
    if not messages:
        messages = [system_prompt, HumanMessage(content=state["question"])]
    else:
        messages = [system_prompt] + messages[1:]

    response = llm.invoke(messages)
    return {"messages": messages + [response]}


# LLMが返した tool_calls を実際の Python関数へルーティングするノード。
# LangGraph 上では「LLMが判断」→「Toolを実行」→「再度LLMへ戻る」という
# エージェント的な流れを学べるのがポイント。
def tool_execution_node(state: ToolCallingState):
    """
    LLMが要求したツールを実行するノード


    Args:
        state: Function Calling用のLangGraph状態


    Returns:
        dict: ToolMessageとtool_traceを含む更新状態
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    tool_trace = list(state.get("tool_trace", []))

    tool_map = {tool_.name: tool_ for tool_ in TOOLS}
    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_result = tool_map[tool_name].invoke(tool_args)

        tool_trace.append(
            {
                "tool": tool_name,
                "args": tool_args,
                "result_preview": str(tool_result)[:300],
            }
        )

        tool_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": messages + tool_messages, "tool_trace": tool_trace}


# Tool Callingが終わった最終メッセージを answer に格納するノード。
# 最後のAIMessageには、ツール利用結果を踏まえた最終回答が入る想定。
def tool_calling_finalize_node(state: ToolCallingState):
    """Tool Calling 完了後の最終回答を状態へ格納する。

    Args:
        state: `ToolCallingState`。

    Returns:
        dict[str, object]: `answer` と `tool_trace` を持つ辞書。
    """
    final_message = state["messages"][-1]
    return {
        "answer": final_message.content,
        "tool_trace": state.get("tool_trace", []),
    }


# 条件分岐関数。
# LLMが tool_calls を返した場合は tool_execution へ進み、
# 返していなければそのまま finalize へ進む。
def should_continue_tool_calling(state: ToolCallingState):
    """Tool Calling の継続可否を判定する。

    Args:
        state: `ToolCallingState`。

    Returns:
        str: `"tool_execution"` または `"finalize"`。
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_execution"

    return "finalize"


# ============================================
# 補助関数：LangGraph作成
# ============================================
def build_rag_graph():
    """通常 RAG 用の LangGraph を構築する。

    Args:
        なし。

    Returns:
        CompiledStateGraph: `rewrite_query -> retrieve -> generate` の流れを持つ実行可能グラフ。
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


# Function Calling 学習用の LangGraph。
# LLMノードと Tool実行ノードを明示的に分けることで、
# Tool Calling の制御フローを理解しやすくしている。
def build_tool_calling_graph():
    """Function Calling 用の LangGraph を構築する。

    Args:
        なし。

    Returns:
        CompiledStateGraph: `agent -> tool_execution -> agent/finalize` の流れを持つ実行可能グラフ。
    """
    graph_builder = StateGraph(ToolCallingState)

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

    return graph_builder.compile()


# LLMによる処理フロー分岐学習用のLangGraph。
# 最初に処理フローを判定し、その結果で後続ノードを切り替える。
def build_workflow_routing_graph():
    """LLM ルーティング用の LangGraph を構築する。

    Args:
        なし。

    Returns:
        CompiledStateGraph: `document` / `web` / `general` に分岐する実行可能グラフ。
    """
    graph_builder = StateGraph(WorkflowRoutingState)

    graph_builder.add_node("classify_workflow_route", classify_workflow_route_node)
    graph_builder.add_node(
        "workflow_document_rewrite_query", workflow_document_rewrite_query_node
    )
    graph_builder.add_node(
        "workflow_document_retrieve", workflow_document_retrieve_node
    )
    graph_builder.add_node(
        "workflow_generate_document_answer", workflow_generate_document_answer_node
    )
    graph_builder.add_node("workflow_web_search", workflow_web_search_node)
    graph_builder.add_node(
        "workflow_generate_web_answer", workflow_generate_web_answer_node
    )
    graph_builder.add_node("workflow_general_answer", workflow_general_answer_node)

    graph_builder.add_edge(START, "classify_workflow_route")

    # classify_workflow_route の結果に応じて、次に進むノードを切り替える
    graph_builder.add_conditional_edges(
        "classify_workflow_route",
        decide_workflow_after_classification,
        {
            "workflow_document_rewrite_query": "workflow_document_rewrite_query",
            "workflow_web_search": "workflow_web_search",
            "workflow_general_answer": "workflow_general_answer",
        },
    )

    graph_builder.add_edge(
        "workflow_document_rewrite_query", "workflow_document_retrieve"
    )
    graph_builder.add_edge(
        "workflow_document_retrieve", "workflow_generate_document_answer"
    )
    graph_builder.add_edge("workflow_generate_document_answer", END)

    graph_builder.add_edge("workflow_web_search", "workflow_generate_web_answer")
    graph_builder.add_edge("workflow_generate_web_answer", END)

    graph_builder.add_edge("workflow_general_answer", END)

    return graph_builder.compile()


# ============================================
# 補助関数：RAG回答生成（Retrieve + Generate）
# ============================================
# RAGの本番処理「検索→回答生成」です。
# 「Retrieve→Generate」の流れを実装。
def answer_question(question, k, prompt_type, chat_history=None):
    """通常 RAG フローで質問に回答する。

    Args:
        question: 現在のユーザー質問。
        k: 検索で取得するチャンク数。
        prompt_type: 回答スタイル。
        chat_history: `list[dict]` 形式の会話履歴。`None` の場合は空履歴として扱う。

    Returns:
        tuple[str, list[tuple[Document, float]]]: 回答文と、検索結果 `(Document, score)` のリスト。
    """
    if st.session_state.rag_graph is None:
        st.session_state.rag_graph = build_rag_graph()

    if st.session_state.persist_dir is None:
        raise ValueError("ベクトルDBが未作成です。先にインデックスを作成してください。")

    initial_state = {
        "question": question,
        "k": k,
        "prompt_type": prompt_type,
        "chat_history": chat_history or [],
        "context_text": "",
        "retrieved_results": [],
        "answer": "",
        "search_query": question,
        "persist_dir": str(st.session_state.persist_dir),
    }

    result = st.session_state.rag_graph.invoke(initial_state)

    return result["answer"], result["retrieved_results"]


# Function Calling経由の回答生成。
# LangGraphがツール選択を含む状態遷移をどう扱うか学べるように、
# 実行ログ(tool_trace)も返して UI に表示できるようにしている。
def answer_question_with_tool_calling(question, prompt_type, chat_history=None):
    """Function Calling RAG フローで質問に回答する。

    Args:
        question: 現在のユーザー質問。
        prompt_type: 回答スタイル。
        chat_history: `list[dict]` 形式の会話履歴。`None` の場合は空履歴として扱う。

    Returns:
        tuple[str, list[dict], list[tuple[Document, float]]]: 回答文、ツール実行ログ、検索結果リスト。
    """
    if st.session_state.tool_calling_graph is None:
        st.session_state.tool_calling_graph = build_tool_calling_graph()

    initial_state = {
        "question": question,
        "prompt_type": prompt_type,
        "chat_history": chat_history or [],
        "answer": "",
        "tool_trace": [],
        "messages": [],
    }

    result = st.session_state.tool_calling_graph.invoke(initial_state)
    return (
        result["answer"],
        result.get("tool_trace", []),
        st.session_state.last_retrieved_results,
    )


# LLMによる処理フロー分岐経由の回答生成。
# 回答に加えて、どの route が選ばれたかも返してUIで可視化する。
def answer_question_with_workflow_routing(question, k, prompt_type, chat_history=None):
    """LLM ルーティング RAG フローで質問に回答する。

    Args:
        question: 現在のユーザー質問。
        k: document フローで取得するチャンク数。
        prompt_type: 回答スタイル。
        chat_history: `list[dict]` 形式の会話履歴。`None` の場合は空履歴として扱う。

    Returns:
        tuple[str, str, list[tuple[Document, float]], str]:
        回答文、選択された route、検索結果、Web 調査メモ。
    """
    # 初回実行時だけ処理フロー分岐用LangGraphを構築して保持する
    if st.session_state.workflow_routing_graph is None:
        st.session_state.workflow_routing_graph = build_workflow_routing_graph()

    initial_state = {
        "question": question,
        "k": k,
        "prompt_type": prompt_type,
        "chat_history": chat_history or [],
        "route": "",
        "context_text": "",
        "retrieved_results": [],
        "answer": "",
        "search_query": question,
        "persist_dir": (
            str(st.session_state.persist_dir) if st.session_state.persist_dir else ""
        ),
        "web_context": "",
    }

    result = st.session_state.workflow_routing_graph.invoke(initial_state)

    return (
        result["answer"],
        result.get("route", ""),
        result.get("retrieved_results", []),
        result.get("web_context", ""),
    )


# ============================================
# 補助関数：共通UI処理
# ============================================
def reset_chat_history():
    """会話履歴・要約メモリ・LangGraph 関連状態をリセットする。

    Args:
        なし。

    Returns:
        None: 会話履歴と関連キャッシュを初期化する。
    """
    st.session_state.chat_history = []

    # 要約メモリも同時にクリアしないと、画面上の履歴と内部メモリがずれる可能性があります。
    if st.session_state.conversation_memory is not None:
        st.session_state.conversation_memory.clear()

    st.session_state.rag_graph = None
    st.session_state.tool_calling_graph = None
    st.session_state.workflow_routing_graph = None
    st.session_state.last_tool_trace = []
    st.session_state.last_workflow_route = ""
    st.session_state.last_web_context = ""


def get_history_for_prompt():
    """設定に応じてプロンプトへ渡す会話履歴を返す。

    Args:
        なし。

    Returns:
        list[dict]: 会話履歴を使う設定なら `st.session_state.chat_history`、使わない設定なら空リスト。
    """
    return st.session_state.chat_history if st.session_state.use_chat_history else []


def update_answer_state(answer, retrieved_results):
    """回答文と検索結果を Session State に反映する。

    Args:
        answer: 最終回答文字列。
        retrieved_results: `list[tuple[Document, float]]` 形式の検索結果。

    Returns:
        None: 回答表示用の Session State を更新する。
    """
    st.session_state.update(
        {
            "last_answer": answer,
            "last_retrieved_results": retrieved_results,
            "last_retrieved_docs": [doc for doc, score in retrieved_results],
        }
    )


def append_chat_history(question, answer):
    """今回の質問と回答を会話履歴へ追加する。

    Notes:
        `ConversationSummaryMemory` が有効な場合は、通常の `chat_history` に加えて
        要約メモリにも同じターンを逐次反映する。

    Args:
        question: ユーザー質問。
        answer: 生成された回答文。

    Returns:
        None: 会話履歴と必要に応じて要約メモリを更新する。
    """
    if st.session_state.use_chat_history:
        st.session_state.chat_history.append({"question": question, "answer": answer})

        # 追加した会話を要約メモリへ逐次反映します。
        # これにより次ターン以降、長い履歴でも summary を参照できます。
        if st.session_state.memory_summary_enabled:
            memory = get_or_create_conversation_memory()
            memory.save_context({"input": question}, {"output": answer})


def render_chunk_preview(chunks):
    """分割済みチャンクのプレビュー UI を描画する。

    Args:
        chunks: `list[Document]` 形式の分割済みチャンク。

    Returns:
        None: Streamlit 上にチャンク内容を表示する。
    """
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
    """会話履歴と要約結果の UI を描画する。

    Args:
        なし。

    Returns:
        None: Streamlit 上に会話履歴を表示する。
    """
    if not st.session_state.chat_history:
        return

    st.subheader("🗂️ 会話履歴")

    # 学習用に、ConversationSummaryMemory がどのように要約しているかを見える化します。
    if st.session_state.memory_summary_enabled:
        with st.expander("📝 ConversationSummaryMemory の要約結果", expanded=False):
            st.text(
                format_chat_history_with_summary(
                    st.session_state.chat_history,
                    max_turns=st.session_state.memory_max_turns_before_summary,
                )
            )
    for i, turn in enumerate(reversed(st.session_state.chat_history), start=1):
        history_index = len(st.session_state.chat_history) - i + 1
        with st.expander(f"会話履歴 #{history_index}"):
            st.markdown(f"**質問**: {turn['question']}")
            st.markdown(f"**回答**: {turn['answer']}")


def render_retrieved_results_view():
    """検索で取得した根拠チャンクの UI を描画する。

    Args:
        なし。

    Returns:
        None: Streamlit 上に検索根拠を表示する。
    """
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


# Function Callingの学習用に、どのツールが呼ばれたかを UI に可視化する。
# これにより「LLM が判断してツールを選んだ」ことを確認しやすくなる。
def render_tool_trace_view():
    """
    Tool Callingの実行ログを表示する


    Returns:
        None
    """
    if not st.session_state.last_tool_trace:
        return

    st.subheader("🛠️ Tool Callingログ")
    for i, trace in enumerate(st.session_state.last_tool_trace, start=1):
        with st.expander(f"Tool #{i}: {trace['tool']}"):
            st.json(
                {
                    "tool": trace["tool"],
                    "args": trace["args"],
                    "result_preview": trace["result_preview"],
                }
            )


# Query Routingの学習用に、どの経路が選ばれたかを表示する。
# 文書検索・Web検索・通常応答のどれに分岐したかを可視化すると、
# ルーティング設計の意図を理解しやすい。
def render_workflow_routing_view():
    """LLM ルーティング結果の UI を描画する。

    Args:
        なし。

    Returns:
        None: Streamlit 上に選択された処理フローを表示する。
    """
    if not st.session_state.last_workflow_route:
        return

    route_labels = {
        "document": "文書検索フロー",
        "web": "Web向けフロー",
        "general": "通常応答フロー",
    }

    st.subheader("🧭 LLMによる処理フロー分岐結果")

    selected_route_label = route_labels.get(
        st.session_state.last_workflow_route,
        st.session_state.last_workflow_route,
    )

    st.info(f"選択された処理フロー: **{selected_route_label}**")

    if (
        st.session_state.last_workflow_route == "web"
        and st.session_state.last_web_context
    ):
        with st.expander("Web調査メモ"):
            st.text(st.session_state.last_web_context)


def render_splitter_comparison(uploaded_file, chunk_size, chunk_overlap):
    """分割方式とパラメータ差分の比較 UI を描画する。

    Args:
        uploaded_file: `st.file_uploader()` が返すアップロードファイル。
        chunk_size: 現在設定のチャンクサイズ。
        chunk_overlap: 現在設定のチャンク重なり幅。

    Returns:
        None: Streamlit 上に比較表とチャンクプレビューを表示する。
    """
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
    """アップロード文書からインデックスを作成し Session State を更新する。

    Args:
        uploaded_file: `st.file_uploader()` が返すアップロードファイル。
        chunk_size: チャンクサイズ。
        chunk_overlap: チャンク重なり幅。

    Returns:
        None: ベクトルストア作成結果を Session State に反映する。
    """
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

            # 2. 古い状態を解除
            st.session_state.rag_graph = None
            st.session_state.tool_calling_graph = None
            st.session_state.workflow_routing_graph = None
            st.session_state.persist_dir = None

            # 3. 分割＆ベクトルDB作成
            _, split_docs, persist_dir = build_vectorstore(
                docs, chunk_size, chunk_overlap, st.session_state.splitter_type
            )

            # 4. 状態更新
            st.session_state.update(
                {
                    "vectorstore_ready": True,
                    "chunks": split_docs,
                    "last_answer": "",
                    "last_retrieved_docs": [],
                    "last_retrieved_results": [],
                    "persist_dir": persist_dir,
                    "last_tool_trace": [],
                    "last_workflow_route": "",
                    "last_web_context": "",
                }
            )

            # 5. 現在使用中以外の古いインデックスを削除
            cleanup_old_persist_dirs(current_persist_dir=persist_dir)

        st.success(f"✅ 完了！チャンク数: **{len(split_docs)}**件")
        st.balloons()  # 成功エフェクト

    except Exception as e:
        st.error(f"❌ エラー: {str(e)}")
        st.caption("ファイル形式やAPIキーを確認してください")


# 回答生成では、通常RAGと Function Calling RAG 等を切り替えられるようにする。
# これにより「検索を固定フローで行う場合」と「LLMがツール判断する場合」の
# 差を同じUIで比較学習できる。
def handle_answer_generation(question):
    """現在の実行モードに応じて回答生成を実行する。

    Args:
        question: ユーザーが入力した質問文。

    Returns:
        None: 回答文・検索結果・ログ類を Session State に反映する。
    """
    if not st.session_state.vectorstore_ready:
        st.warning("⚠️ まず「インデックス作成」を実行してください")
        return
    if not question.strip():
        st.warning("⚠️ 質問を入力してください")
        return
    if not check_api_key():
        st.error("❌ .env の OPENAI_API_KEY を確認")
        return

    try:
        with st.spinner("検索中...回答生成中..."):
            history_for_prompt = get_history_for_prompt()

            if st.session_state.execution_mode == "Function Calling RAG":
                answer, tool_trace, retrieved_results = (
                    answer_question_with_tool_calling(
                        question,
                        st.session_state.prompt_type,
                        history_for_prompt,
                    )
                )
                st.session_state.last_tool_trace = tool_trace
                st.session_state.last_workflow_route = ""
                st.session_state.last_web_context = ""

            elif st.session_state.execution_mode == "LLMルーティング RAG":
                answer, route, retrieved_results, web_context = (
                    answer_question_with_workflow_routing(
                        question,
                        st.session_state.retrieval_k,
                        st.session_state.prompt_type,
                        history_for_prompt,
                    )
                )
                st.session_state.last_tool_trace = []
                st.session_state.last_workflow_route = route
                st.session_state.last_web_context = web_context

            else:
                answer, retrieved_results = answer_question(
                    question,
                    st.session_state.retrieval_k,
                    st.session_state.prompt_type,
                    history_for_prompt,
                )
                st.session_state.last_tool_trace = []
                st.session_state.last_workflow_route = ""
                st.session_state.last_web_context = ""

            update_answer_state(answer, retrieved_results)
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
            600,
            100,
            help="1チャンクの文字数（大きい=文脈多め、小さい=細かい検索）",
        )

    with col2:
        chunk_overlap = st.slider(
            "🔗 重なり幅", 0, 300, 60, 20, help="チャンク間の重なり（文脈切れを防ぐ）"
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

    execution_mode = st.radio(
        "⚙️ 実行モード",
        MODE_TYPES,
        index=MODE_TYPES.index(st.session_state.execution_mode),
        help=(
            "通常RAGは固定フロー、Function Calling RAGはLLMが必要に応じてツールを選び、"
            "LLMルーティング RAGは質問内容ごとに最適な処理フローへ分岐します。"
        ),
    )

    st.session_state.prompt_type = prompt_type
    st.session_state.retrieval_k = retrieval_k
    st.session_state.execution_mode = execution_mode

    # 単発RAGと対話型RAGを切り替えるUI
    use_chat_history = st.checkbox(
        "🧠 会話履歴を使う",
        value=st.session_state.use_chat_history,
        help="オンにすると、前の質問と回答を参考にして次の質問へつなげます。",
    )

    st.session_state.use_chat_history = use_chat_history

    # 長い会話を summary に圧縮するかどうかを切り替えるUIです。
    memory_summary_enabled = st.checkbox(
        "📝 ConversationSummaryMemoryで長い履歴を要約する",
        value=st.session_state.memory_summary_enabled,
        help="オンにすると、過去会話を要約しつつ直近の数ターンはそのまま保持します。",
    )
    st.session_state.memory_summary_enabled = memory_summary_enabled

    # 何ターン分を「生の履歴」として残すかを調整します。
    memory_max_turns_before_summary = st.slider(
        "🧾 要約とは別に保持する直近ターン数",
        min_value=1,
        max_value=10,
        value=st.session_state.memory_max_turns_before_summary,
        step=1,
        help="要約だけでなく、そのまま参照させたい直近の会話ターン数です。",
    )
    st.session_state.memory_max_turns_before_summary = memory_max_turns_before_summary

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
        st.caption(
            "使用プロンプト: "
            f"{st.session_state.prompt_type} / "
            f"実行モード: {st.session_state.execution_mode}"
        )
        st.markdown(f"**{st.session_state.last_answer}**")

    render_tool_trace_view()
    render_workflow_routing_view()

    # 現在の会話履歴を表示
    render_chat_history_view()

    # 検索根拠表示
    render_retrieved_results_view()

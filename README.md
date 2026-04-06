# langchain_rag_tutorial_app

LangChain と Streamlit を使って、RAG（Retrieval-Augmented Generation）の基本を学べる初心者向けチュートリアルアプリです。

このアプリでは、文書のアップロード、チャンク分割、ベクトルDBへの保存、関連チャンクの検索、回答生成までの流れを、UIで確認しながら体験できます。

## デモ画面
- webアプリ立ち上げ後は、RAG及びLangChainの概要を表示

<p align="center">
  <img src="images/RAG概要.png" alt="LangChain RAG Tutorial App1" width="900">
</p>

- txtもしくはmd形式のファイル入力に対応しており、入力文章に対する質問を返してくれる

<p align="center">
  <img src="images/UIイメージ.png" alt="LangChain RAG Tutorial App2" width="900">
</p>

### 回答例
- 作成したチュートリアルアプリの回答精度の検証を目的として、2つの文章を用意して回答精度を確認

1. 架空企業の社内ハンドブックを作成し、期待通りの回答を得られれることを確認
    - パスワードの条件を質問し、社内ハンドブックに記載の情報を取得成功

<p align="center">
  <img src="images/回答例1.png" alt="社内ハンドブック回答例" width="900">
</p>

2. 架空製品マニュアルを作成し、期待通りの回答を得られれることを確認
    - 録画データの保存先を質問し、製品マニュアル記載の情報を取得成功

<p align="center">
  <img src="images/回答例2.png" alt="製品回答例" width="900">
</p>

## 特徴

- LangChain を使った RAG の基本フローを学べる
- Streamlit でローカル実行しやすい
- 初学者向けの解説付き UI
- 検索されたチャンクを確認できる
- `.txt` と `.md` ファイルを対象にしたシンプルな構成

## ディレクトリ構成

```bash
langchain_rag_tutorial_app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
├── .env.example
└── data/
```

## 使用技術

- Streamlit
- LangChain
- langchain-openai
- langchain-community
- langchain-text-splitters
- ChromaDB
- python-dotenv

## 前提条件

- Python 3.10 以上を推奨
- OpenAI API キー

## セットアップ

### 1. リポジトリを作成またはクローン

```bash
git clone <your-repository-url>
cd langchain_rag_tutorial_app
```

### 2. 仮想環境を作成

```bash
python -m venv .venv
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 3. パッケージをインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数を設定

`.env.example` をコピーして `.env` を作成し、OpenAI API キーを設定してください。

```bash
cp .env.example .env
```

`.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 実行方法

```bash
streamlit run app.py
```

起動後、ブラウザで表示されるローカルURLにアクセスしてください。

## アプリで学べること

1. 文書を読み込む
2. 文書をチャンクに分割する
3. 埋め込みを作成する
4. ベクトルDBに保存する
5. 質問に近いチャンクを検索する
6. 検索結果をもとに回答を生成する

## 注意点

- `.env` は機密情報を含むため、GitHub に push しないでください
- 初回実行時は埋め込み作成とベクトル化に少し時間がかかる場合があります
- LangChain のバージョンによって API 差分が出ることがあるため、`requirements.txt` は固定するのがおすすめです

## 今後の改善案

- PDF アップロード対応
- 回答時の参照元表示強化
- 会話履歴の保持
- 複数ファイル対応
- Ollama などローカルLLM対応

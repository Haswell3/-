import openai
from django.shortcuts import render, redirect
from .forms import AudioUploadForm, TextFileUploadForm
from django.http import HttpResponse
import os
from datetime import datetime
from django.core.files import File

from pydub import AudioSegment
from django.conf import settings
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import Chroma, faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter #LangChain関係のインポート
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
#from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
from typing import Any
import math

openai.api_key = ""

#句読点で分割させるコード
class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata

def split_audio(audio_segment, segment_length_ms=600000):  # 10分間のセグメント
    duration_ms = len(audio_segment)
    segments_count = math.ceil(duration_ms / segment_length_ms)
    segments = []

    for i in range(segments_count):
        start_ms = i * segment_length_ms
        end_ms = min((i + 1) * segment_length_ms, duration_ms)
        segment = audio_segment[start_ms:end_ms]
        segments.append(segment)
    return segments

def home(request):
    return render(request, "home.html")


def upload_audio(request):
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            audio_file_path = instance.audio.path  # アップロードされたファイルのパスを取得
            file_extension = audio_file_path.lower().split('.')[-1]

            # ファイルの拡張子に応じて処理を分岐
            if file_extension in ['wav', 'mp3']:
                # オーディオファイルを読み込む
                sound = AudioSegment.from_file(audio_file_path, format=file_extension)
                segments = split_audio(sound)

                transcripts = []
                for i, segment in enumerate(segments):
                    # WAVファイルの場合はMP3に変換
                    segment_format = 'mp3' if file_extension == 'wav' else file_extension
                    segment_path = os.path.join(settings.MEDIA_ROOT, f"segment_{i}.{segment_format}")
                    segment.export(segment_path, format=segment_format)

                    with open(segment_path, "rb") as audio_file:
                        # ここでOpenAIの音声認識APIを呼び出す
                        transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt="こんにちは、お願いします。")
                        transcripts.append(transcript["text"])


                    # 中間ファイルを削除
                    os.remove(segment_path)

                full_transcript = " ".join(transcripts)

                # 現在の日時を取得して、ファイル名を生成
                current_time = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"transcript_{current_time}.txt"

                # 'output' ディレクトリの存在をチェックし、存在しない場合は作成
                output_directory = os.path.join(settings.MEDIA_ROOT, 'output')
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # ファイルのフルパスを生成
                filepath = os.path.join(output_directory, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(full_transcript)

                # ファイルをDjangoのFileオブジェクトに変換して保存
                with open(filepath, 'rb') as f:
                    instance.transcript = File(f, name=filename)
                    instance.save()

                # 処理が完了したらリダイレクト
                return redirect('upload_text')

            else:
                # WAVまたはMP3以外のファイル形式の場合はエラーを返す
                return HttpResponse("wavか、mp3以外はサポートしていません。", status=400)

    else:
        form = AudioUploadForm()
    return render(request, 'upload.html', {'form': form})
def upload_text(request):
    if request.method == 'POST':
        form = TextFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            # ファイルのパスを直接使用してファイルを開きます
            text_file_path = instance.text_file.path
            with open(text_file_path, "r", encoding='utf-8') as text_file:
                text_content = text_file.read()

                # テキストを分割
                japanese_spliter = JapaneseCharacterTextSplitter(chunk_size=1000, chunk_overlap=5) 
                #chunk_size=はchunkの文字数、chunk_overlap=はchunk同士を何文字被らせるか
                
                chunks = japanese_spliter.split_text(text_content)
                print(chunks)
                # ドキュメントオブジェクトのリストを作成する
                documents = [Document(chunk, metadata={'additional_info': 'your_data_here'}) for chunk in chunks]

                # LangChainのセットアップ
                embeddings = OpenAIEmbeddings(openai_api_key="")
                vectorstore = Chroma.from_documents(documents, embeddings)
                memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
                # APIキーを直接渡す
                qa_chain = ConversationalRetrievalChain.from_llm(
                    ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=4500, openai_api_key=""),
                    vectorstore.as_retriever(search_kwargs={"k": 1}),
                    memory=memory
                )

                # クエリの実行
                query = "議事録を議事、決定事項、Todoを下記のフォーマットで整理してください。■議事 ■決定事項 ■TODO"
                # query = "議事録を作成してください"
                result = qa_chain({"question": query})
                answer = result["answer"]
                print(answer)

                # ファイルに保存
                current_time = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"ToDo_{current_time}.txt"
                output_directory = 'output'
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                filepath = os.path.join(output_directory, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(answer)

            return redirect('upload_text')
    else:
        form = TextFileUploadForm()
    return render(request, 'rag.html', {'form': form})

    

# import openai
# from pyannote.audio import Pipeline
# import torch
# from django.shortcuts import render, redirect
# from .forms import AudioUploadForm
# from django.http import HttpResponse

# openai.api_key = ""
# HUGGINGFACE_ACCESS_TOKEN = ""

# try:
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.0",
#         use_auth_token=HUGGINGFACE_ACCESS_TOKEN)
#     pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# except Exception as e:
#     print(f"Error loading pipeline: {e}")
#     pipeline = None

# def home(request):
#     return HttpResponse("Hello, Django!")

# def upload_audio(request):
#     if request.method == 'POST':
#         form = AudioUploadForm(request.POST, request.FILES)
#         if form.is_valid() and pipeline:
#             instance = form.save()
#             with instance.audio.open("rb") as audio_file:
#                 audio_path = audio_file.name
#                 diarization = pipeline(audio_path)
#                 for turn, _, speaker in diarization.itertracks(yield_label=True):
#                     # Extract the audio segment based on turn.start and turn.end
#                     # NOTE: The extraction code will depend on the method or tool you use
#                     segment = ...  
                    
#                     # Transcribe each segment with OpenAI's whisper
#                     # NOTE: This is a sample and might require modifications based on the actual method
#                     transcript = openai.Audio.transcribe("whisper-1", segment)
#                     with open('output.txt', 'a', encoding='utf-8') as f:
#                         f.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}: {transcript['text']}\n")
#             return redirect('upload_audio')
#     else:
#         form = AudioUploadForm()
#     return render(request, 'upload.html', {'form': form})

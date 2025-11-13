import sys
import os
import gradio as gr
from functools import lru_cache
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 파일에서 API 키 로드
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    sys.exit(1)


@lru_cache(maxsize=1)
def get_llm():
    """LLM 초기화 - 앱 실행 시 한 번만 생성"""
    print("🤖 LLM을 초기화합니다... (OpenAI: gpt-4o-mini)")
    return ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=2000,
        temperature=0.7,
        streaming=True,
    )


@lru_cache(maxsize=4) # PDF 경로별로 캐시
def get_retriever_from_pdf(pdf_path: str):
    """PDF 파일로부터 Knowledge Base Retriever 초기화"""
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        # Gradio 인터페이스에서는 오류를 raise하는 것이 좋음
        raise gr.Error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    
    try:
        print(f"📚 PDF 문서를 로드합니다: {pdf_path}")
        # 1. 문서 로드
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise gr.Error(f"PDF에서 문서를 로드하지 못했습니다: {pdf_path}")

        # 2. 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. 임베딩 및 벡터 스토어 생성
        print("🧠 문서를 임베딩하고 벡터 스토어를 생성합니다... (FAISS)")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(splits, embeddings)

        # 4. Retriever 반환
        return vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
    except Exception as e:
        print(f"❌ Retriever 초기화 실패: {e}")
        raise gr.Error(f"Retriever 초기화 실패: {str(e)}")


def format_docs(docs):
    """검색된 문서를 문자열로 변환 (LangChain Document 객체용)"""
    if not docs:
        print("⚠️ 검색된 문서 없음")
        return "관련 문서를 찾을 수 없습니다."

    formatted = [
        doc.page_content for doc in docs if hasattr(doc, "page_content") and doc.page_content
    ]

    result = (
        "\n\n---\n\n".join(formatted)
        if formatted
        else "문서 내용을 추출할 수 없습니다."
    )
    print(f"✅ {len(formatted)}개 문서 포맷 완료 (총 {len(result)}자)")
    return result


def create_chain_with_kb(retriever, llm):
    """RAG 체인 생성 - Retriever로 문서 검색 후 LLM에 전달"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """다음 문맥(context)을 참고하여 질문에 답변하세요.
문맥에 답이 없으면 모른다고 답하세요.

Context:
{context}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    def retrieve_and_format(x):
        """검색 실행 및 포맷팅"""
        try:
            input_text = x["input"] if isinstance(x, dict) else x
            print(f"\n🔍 검색 쿼리: '{input_text}'")
            retrieved_docs = retriever.invoke(input_text)
            print(f"📊 검색 결과: {len(retrieved_docs) if retrieved_docs else 0}개")
            return format_docs(retrieved_docs)
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return f"검색 중 오류 발생: {str(e)}"

    return (
        {
            "context": retrieve_and_format,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
    )


def create_chain_without_kb(llm):
    """일반 대화용 체인 - KB 없이 LLM만 사용"""
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm


# --- Gradio UI 및 채팅 로직 ---

llm = get_llm() # LLM은 앱 시작 시 한 번만 로드

def chat_response(user_input, chat_history_list, use_kb, pdf_path):
    """
    Gradio의 Chatbot UI와 연동되는 메인 함수.
    user_input: 사용자의 새 입력 (str)
    chat_history_list: Gradio 챗봇의 대화 기록 (List[List[str, str]])
    use_kb: KB 사용 여부 (bool)
    pdf_path: PDF 파일 경로 (str)
    """
    
    # 1. Gradio의 대화 기록을 LangChain 형식으로 변환
    langchain_history = []
    for user_msg, ai_msg in chat_history_list:
        if user_msg:
            langchain_history.append(HumanMessage(content=user_msg))
        if ai_msg:
            langchain_history.append(AIMessage(content=ai_msg))
    
    # 2. 체인 선택
    try:
        if use_kb:
            if not pdf_path:
                raise gr.Error("Knowledge Base를 사용하려면 PDF 파일 경로를 입력해야 합니다.")
            # 캐시된 Retriever 가져오기
            retriever = get_retriever_from_pdf(pdf_path)
            chain = create_chain_with_kb(retriever, llm)
            print("ℹ️ RAG 모드로 응답합니다.")
        else:
            chain = create_chain_without_kb(llm)
            print("ℹ️ 일반 대화 모드로 응답합니다.")
            
    except Exception as e:
        # get_retriever_from_pdf에서 오류 발생 시
        yield chat_history_list + [[user_input, str(e)]]
        return

    # 3. 스트리밍 응답 처리
    full_response = ""
    # UI에 사용자 메시지 먼저 표시
    chat_history_list.append([user_input, ""])
    
    try:
        # streaming
        for chunk in chain.stream(
            {
                "chat_history": langchain_history, # 현재 입력을 제외한 이전 기록
                "input": user_input,
            }
        ):
            full_response += chunk.content
            # 마지막 메시지(AI 응답)를 실시간으로 업데이트
            chat_history_list[-1][1] = full_response
            yield chat_history_list
            
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        print(f"❌ {error_msg}")
        chat_history_list[-1][1] = error_msg
        yield chat_history_list


def clear_chat():
    """채팅 기록을 지우는 함수"""
    return [], []

def update_status_text(use_kb, pdf_path):
    """KB 상태 표시줄 업데이트"""
    if use_kb:
        if pdf_path and os.path.exists(pdf_path):
            return f"✅ KB 사용 중 ({os.path.basename(pdf_path)})"
        elif pdf_path:
            return f"⚠️ KB 사용 체크됨 (파일 경로 확인 필요)"
        else:
            return f"⚠️ KB 사용 체크됨 (PDF 경로 없음)"
    else:
        return "ℹ️ 일반 대화 모드"

# --- Gradio UI 레이아웃 ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Chatbot with PDF Knowledge Base (Gradio)")

    with gr.Row():
        # --- 1. 사이드바 (Streamlit의 sidebar와 유사) ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## ⚙️ 설정")
            
            pdf_path_input = gr.Textbox(
                label="PDF 파일 경로",
                value="nov1025.pdf", # 기본값 설정
                info="RAG에 사용할 PDF 파일의 경로를 입력하세요.",
            )
            use_kb_checkbox = gr.Checkbox(
                label="Knowledge Base 사용", 
                value=True
            )
            status_display = gr.Markdown(
                update_status_text(True, "nov1025.pdf") # 초기 상태
            )
            
            # 설정 변경 시 상태 텍스트 업데이트
            use_kb_checkbox.change(fn=update_status_text, inputs=[use_kb_checkbox, pdf_path_input], outputs=status_display)
            pdf_path_input.change(fn=update_status_text, inputs=[use_kb_checkbox, pdf_path_input], outputs=status_display)

        # --- 2. 메인 챗봇 창 ---
        with gr.Column(scale=4):
            chatbot_display = gr.Chatbot(
                label="Chat",
                height=600,
                bubble_full_width=False
            )
            user_input_textbox = gr.Textbox(
                placeholder="Message OpenAI...",
                label="User Input",
                scale=7
            )
            clear_btn = gr.Button("🗑️ Clear Chat")

    # --- 3. 이벤트 핸들러 연결 ---
    
    # 사용자 입력(submit) 시
    user_input_textbox.submit(
        fn=chat_response,
        inputs=[user_input_textbox, chatbot_display, use_kb_checkbox, pdf_path_input],
        outputs=[chatbot_display]
    ).then(
        fn=lambda: "", # 입력창 비우기
        inputs=None,
        outputs=user_input_textbox
    )

    # 채팅 클리어 버튼 클릭 시
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[user_input_textbox, chatbot_display]
    )

# --- 앱 실행 ---
if __name__ == "__main__":
    print("Gradio 앱을 실행합니다. http://127.0.0.1:7860 에서 확인하세요.")
    demo.launch()

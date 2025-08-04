from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os, fitz, uuid, json, logging, tiktoken, shutil
import datetime

timestamp = datetime.datetime.now().isoformat()

load_dotenv()
app = FastAPI()

PDF_DIR = "local_pdfs"
VECTOR_STORE_PATH = "vector_db"
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.json")
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MAX_TOKEN_LIMIT = 300, 50, 3, 14000

Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)

log_path = "logs/ask_logs.log"
Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

vector_store = None


def count_tokens(text: str) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(data):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def init_vector_store():
    global vector_store
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        try:
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        except:
            vector_store = None
    else:
        vector_store = None


def reload_vector_store():
    init_vector_store()


init_vector_store()


def process_pdf(filepath, filename):
    global vector_store

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    with fitz.open(filepath) as doc:
        text = "\n".join([page.get_text() for page in doc])

    docs = splitter.create_documents([text])
    file_id = str(uuid.uuid4())

    for doc in docs:
        doc.metadata = {"file_id": file_id, "filename": filename}

    if vector_store:
        vector_store.add_documents(docs)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)

    vector_store.save_local(VECTOR_STORE_PATH)
    metadata = load_metadata()
    metadata[filename] = file_id
    save_metadata(metadata)


@app.post("/api/v1/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = file.filename
        filepath = os.path.join(PDF_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        process_pdf(filepath, filename)
        reload_vector_store()

        return JSONResponse(
            content={"status": "success", "message": f"فایل {filename} با موفقیت آپلود و پردازش شد."},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": "خطا در بارگذاری فایل", "detail": str(e)},
            status_code=500
        )


class Question(BaseModel):
    question: str
    recentQuestions: List[str] = []


@app.post("/api/v1/ask")
async def ask_question(q: Question):
    # if not vector_store:
    #     return JSONResponse(content={"status": "error", "message": "هیچ سندی وجود ندارد."}, status_code=503)

    try:
        # فیلتر سؤالات عمومی
        عمومی = [
            "لاراول چیست", "ماشین چیست", "انسان چیست", "هوا چیست", "آب چیست",
            "برنامه نویسی چیست", "کامپیوتر چیست"
        ]
        lowered_q = q.question.strip().lower()
        if any(term in lowered_q for term in عمومی):
            return JSONResponse(
                content={"status": "out_of_scope",
                         "message": "این سوال در حوزه تخصصی ما نیست. لطفاً سؤال مرتبط با حوزه مالیاتی بپرسید."},
                status_code=400
            )

        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        relevant_docs = retriever.get_relevant_documents(q.question)

        filtered_docs = [doc for doc in relevant_docs if isinstance(doc.page_content, str)]

        # اگر در حوزه مالیاتی است ولی جواب نداریم
        if not filtered_docs:
            if "مالیات" in lowered_q or "مالیاتی" in lowered_q:
                return JSONResponse(
                    content={
                        "status": "not_found",
                        "message": "برای رسیدن به جواب این سوال لطفاً با این شماره ۴۱۲۶۷۰۰۰ با پشتیبانی تماس بگیرید یا از طریق تیکت به ادرس https://pazhtsp.ir/about-us/ با ما در ارتباط باشید"
                    },
                    status_code=404
                )
            else:
                return JSONResponse(
                    content={"status": "not_found", "message": "پاسخی برای این سوال یافت نشد."},
                    status_code=404
                )

        combined_text = "\n\n".join([doc.page_content for doc in filtered_docs])
        if count_tokens(combined_text + q.question) > MAX_TOKEN_LIMIT:
            return JSONResponse(content={"status": "too_large", "message": "لطفا در سوال خود یک بازنگری بفرمایید."}, status_code=413)

        recent_prompt = "سوالات قبلی:\n" + "\n".join([f"{i + 1}. {s.strip()}" for i, s in enumerate(q.recentQuestions)])
        full_prompt = f"{recent_prompt}\n\nسوال جدید:\n{q.question.strip()}" if q.recentQuestions else q.question.strip()

        # 📝 پرامپت سفارشی
        system_instructions = """
        شما یک دستیار هوشمند که اسمت معتمند هوشمند پاژ هست که در حوزه کارگزار متخصص و متعهد در امور مالیاتی و اجرای پایانه های فروشگاهی و سامانه مودیان هستید.
        فقط به سؤالات مرتبط با مالیات و حوزه کاری شرکت پاسخ دهید.
        از پاسخ‌های خشک و کلیشه‌ای مانند "در متنی که در دسترس است..." یا "اطلاعاتی که شما بیان کردید به من مراجعه نمی‌کند" استفاده نکنید.
        اگر جواب دقیقی ندارید، محترمانه بگویید در داده‌های فعلی نیست و کاربر را به پشتیبانی که شماره اش ۴۱۲۶۷۰۰۰ یا تیکت https://pazhtsp.ir/about-us/ ارجاع دهید.
        در پایان پاسخ، کاربر را تشویق کنید که اگر سؤال دیگری دارد بپرسد.
        لطفاً همیشه جواب‌ها را در قالب HTML برگردان ولی بدون تگ html در حالی که شماره تلقن یا ادرس سایتی به صورت لینک و بولد باشن.
        """

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": None},
            return_source_documents=False
        )

        # مدل OpenAI رو با دستور بالا صدا می‌زنیم با دستورالعمل
        answer = llm.predict(f"""{system_instructions}\n\nمتن مرجع:\n{combined_text}\n\nسؤال:\n{full_prompt}""")
        # logging.info(f"""{system_instructions}\n\nمتن مرجع:\n{combined_text}\n\nسؤال:\n{full_prompt}""")

        if not answer or len(answer.strip()) < 3:
            return JSONResponse(content={"status": "empty_answer", "message": "پاسخی یافت نشد."})

        return JSONResponse(content={"status": "success", "answer": answer}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": "خطا در پردازش سؤال", "detail": str(e)},
                            status_code=500)


@app.delete("/api/v1/delete-pdf/{filename}")
async def delete_pdf(filename: str):
    global vector_store
    try:
        filepath = os.path.join(PDF_DIR, filename)
        metadata = load_metadata()

        if filename not in metadata:
            return JSONResponse(content={"status": "not_found", "message": "فایل یافت نشد."}, status_code=404)

        file_id = metadata[filename]

        if not vector_store or not hasattr(vector_store, "docstore"):
            return JSONResponse(content={"status": "error", "message": "vector_store به درستی بارگذاری نشده."},
                                status_code=500)

        keys_to_remove = [k for k, d in vector_store.docstore._dict.items() if d.metadata.get("file_id") == file_id]
        for k in keys_to_remove:
            vector_store.docstore._dict.pop(k, None)

        vector_store.index_to_docstore_id = {
            k: v for k, v in vector_store.index_to_docstore_id.items() if v not in keys_to_remove
        }

        del metadata[filename]
        save_metadata(metadata)

        from langchain_community.vectorstores.faiss import dependable_faiss_import
        faiss = dependable_faiss_import()

        texts = [doc.page_content for doc in vector_store.docstore._dict.values()]
        metadatas = [doc.metadata for doc in vector_store.docstore._dict.values()]

        if texts:
            new_vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            new_vector_store.save_local(VECTOR_STORE_PATH)
            vector_store = new_vector_store
        else:
            shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            vector_store = None

        if os.path.exists(filepath):
            os.remove(filepath)

        reload_vector_store()
        return JSONResponse(
            content={"status": "success", "message": f"فایل {filename} و امبدهای مرتبط حذف شدند."},
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": "خطا در حذف فایل", "detail": str(e)},
            status_code=500
        )


@app.put("/api/v1/update-pdf/{filename}")
async def update_pdf(filename: str, file: UploadFile = File(...)):
    try:
        await delete_pdf(filename)
        filepath = os.path.join(PDF_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())
        process_pdf(filepath, filename)
        reload_vector_store()
        return JSONResponse(
            content={"status": "success", "message": f"فایل {filename} با موفقیت به‌روزرسانی شد."},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": "خطا در به‌روزرسانی فایل", "detail": str(e)},
            status_code=500
        )

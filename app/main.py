from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil

from app.rag import ask_question
from app.ingest import ingest_file

app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):

    answer = ask_question(q.question)

    return {"question": q.question, "answer": answer}


UPLOAD_FOLDER = "documents"


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_file(file_path)

    return {"status": "document indexed"}

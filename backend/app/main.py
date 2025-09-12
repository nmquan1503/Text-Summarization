from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from .core.model_wrapper import ModelWrapper
from .config import (
    MODEL_CONFIG_PATH,
    MODEL_WEIGTHS_PATH,
    VOCAB_PATH,
    MAX_DOC_LENGTH,
    MAX_SENT_LENGTH,
    MAX_OUTPUT_LENGTH,
    BEAM_SIZE
)

wrapper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global wrapper
    wrapper = ModelWrapper(
        vocab_path=VOCAB_PATH,
        model_config_path=MODEL_CONFIG_PATH,
        model_weights_path=MODEL_WEIGTHS_PATH,
        max_doc_length=MAX_DOC_LENGTH,
        max_sent_length=MAX_SENT_LENGTH,
        max_output_length=MAX_OUTPUT_LENGTH,
        beam_size=BEAM_SIZE
    )
    yield
    wrapper = None

app = FastAPI(title='Vietnamese Text Summarization API', lifespan=lifespan)
origins = [
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    text: str

class Response(BaseModel):
    summary: str

@app.post('/summarize', response_model=Response)
def summarize(request: Request):
    if not request.text.strip():
        return Response(summary="")
    
    try:
        summary = wrapper.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return Response(summary=summary)
"""Medical chat assistant router."""

from fastapi import APIRouter, Depends, File, UploadFile

from medvision.domain.entities.user import User
from medvision.infrastructure.llm import LLMService
from medvision.presentation.api.dependencies import get_current_user
from medvision.presentation.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])
llm_service = LLMService()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _user: User = Depends(get_current_user),
):
    if request.use_rag:
        response = llm_service.answer_medical_question(request.message, use_rag=True)
        if hasattr(response, "citations"):
            return ChatResponse(
                answer=response.answer,
                citations=[
                    {"source": c.source, "page": c.page, "chunk_id": c.chunk_id}
                    for c in response.citations
                ],
            )
    answer = llm_service.answer_medical_question(request.message, use_rag=False)
    return ChatResponse(answer=str(answer))


@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    _user: User = Depends(get_current_user),
):
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    num_chunks = llm_service.rag.ingest_pdf(tmp_path, source_name=file.filename)
    tmp_path.unlink(missing_ok=True)
    return {"filename": file.filename, "chunks_ingested": num_chunks}

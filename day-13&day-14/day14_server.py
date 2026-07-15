import logging
from typing import List, Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from day13_generator import RAGContextAssembler

logger = logging.getLogger(__name__)


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    context_chunks: List[str] = Field(default_factory=list, max_length=20)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be empty or whitespace.")
        return value


class RAGResponse(BaseModel):
    compiled_prompt: str
    status: Literal["success"]


app = FastAPI(title="Tokyo AI/ML Engineering Sprint - Day 14 Integration Engine")
assembler = RAGContextAssembler(max_chars=4000)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post(
    "/api/v1/assemble", response_model=RAGResponse, status_code=status.HTTP_200_OK
)
async def serve_context_assembly(payload: RAGRequest):
    try:
        compiled_prompt = assembler.assemble_prompt(
            query=payload.query,
            context_chunks=payload.context_chunks,
        )
        return RAGResponse(compiled_prompt=compiled_prompt, status="success")

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception:
        logger.exception("Unexpected assembly failure")

        raise HTTPException(
            status_code=500, detail="Internal core assembly error occurred."
        )

from fastapi import APIRouter, Depends
from app.services import similarity_service
from pydantic import BaseModel

router = APIRouter(prefix="/similarity", tags=["similarity"])

class SentenceSimilarityRequest(BaseModel):
        sentence1: str
        sentence2: str

@router.post("/jaccard")
def calculate_jaccard(request: SentenceSimilarityRequest, service: similarity_service.SimilarityService = Depends()):
    similarity = service.calculate_jaccard_similarity(request.sentence1, request.sentence2)
    return {"jaccard_similarity": similarity}

@router.post("/cosine_tfidf")
def calculate_cosine_tfidf(request: SentenceSimilarityRequest, service: similarity_service.SimilarityService = Depends()):
    similarity = service.calculate_cosine_similarity_tfidf(request.sentence1, request.sentence2)
    return {"cosine_similarity_tfidf": similarity}
from fastapi import FastAPI
from app.routes import similarity

app = FastAPI()

app.include_router(similarity.router)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}


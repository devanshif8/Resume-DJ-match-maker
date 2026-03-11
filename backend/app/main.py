from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.analyer import router as analyze_router

app = FastAPI(title="Resume Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Resume Matcher API is running"}
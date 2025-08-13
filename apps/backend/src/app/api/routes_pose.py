from typing import Annotated

from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def camera():
  return {"message": "Hello World"}
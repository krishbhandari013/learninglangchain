import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

# ==================================================
# API KEYS
# ==================================================

load_dotenv()

# --------------------------------------------------
# DEFINE STRUCTURED OUTPUT SCHEMA
# --------------------------------------------------
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    release_year: int = Field(description="Release year")
    genre: List[str] = Field(description="Genres")
    director: str = Field(description="Director name")
    rating: float = Field(description="IMDb or approximate rating out of 10")
    language: str = Field(description="Primary language")
    duration: str = Field(description="Movie duration")
    review: str = Field(description="Short review summary")

# --------------------------------------------------
# CREATE GROQ MODEL
# --------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# --------------------------------------------------
# ENABLE STRUCTURED OUTPUT
# --------------------------------------------------
structured_llm = llm.with_structured_output(MovieReview)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a movie expert.

Given a movie name, provide:

- title
- release_year
- genre
- director
- rating
- language
- duration
- short review

Movie: {movie}
"""
)

# --------------------------------------------------
# CHAIN
# --------------------------------------------------
chain = prompt | structured_llm

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
movie_name = input("Enter movie name: ")

result = chain.invoke({
    "movie": movie_name
})

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
print("\n=== Movie Details ===\n")

print(f"Title        : {result.title}")
print(f"Release Year : {result.release_year}")
print(f"Genre        : {', '.join(result.genre)}")
print(f"Director     : {result.director}")
print(f"Rating       : {result.rating}/10")
print(f"Language     : {result.language}")
print(f"Duration     : {result.duration}")
print(f"Review       : {result.review}")

print("\n=== JSON Output ===\n")
print(result.model_dump_json(indent=2))
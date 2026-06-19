from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

# ==================================================
# API KEYS
# ==================================================

load_dotenv()
# -----------------------------
# 2. SCHEMA (YOUR VERSION)
# -----------------------------

class Director(BaseModel):
    name: str = Field(description="Full name of the director")
    nationality: str = Field(description="Director's nationality")

class Actor(BaseModel):
    name: str = Field(description="Actor's full name")
    role: str = Field(description="Role played in the movie")

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    release_year: int = Field(description="Year the movie was released")
    genre: List[str] = Field(description="List of genres like Action, Drama, Sci-Fi")

    rating: float = Field(description="IMDb rating out of 10")
    language: str = Field(description="Primary language of the movie")
    duration: str = Field(description="Movie runtime (e.g., 2h 30m)")

    director: Director = Field(description="Director details")
    cast: List[Actor] = Field(description="Main cast of the movie")

    budget: Optional[str] = Field(description="Estimated production budget")
    box_office: Optional[str] = Field(description="Total box office collection")

    country: str = Field(description="Country where movie was produced")
    production_company: Optional[str] = Field(description="Production company name")

    plot_summary: str = Field(description="Short summary of the movie story")
    review: str = Field(description="AI-generated review of the movie")
    recommendation: str = Field(description="Whether user should watch it or not")

    awards: Optional[List[str]] = Field(description="List of awards won by the movie")

# -----------------------------
# 3. LLM (GROQ)
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -----------------------------
# 4. STRUCTURED OUTPUT WRAPPER
# -----------------------------

structured_llm = llm.with_structured_output(MovieReview)

# -----------------------------
# 5. PROMPT
# -----------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a movie expert.

Given a movie name, generate complete structured information:

Movie: {movie}

Fill all fields:
- title
- release_year
- genre
- rating
- language
- duration
- director (name, nationality)
- cast (name, role)
- budget (if known)
- box_office (if known)
- country
- production_company
- plot_summary
- review
- recommendation
- awards (if any)
"""
)

# -----------------------------
# 6. CHAIN
# -----------------------------

chain = prompt | structured_llm

# -----------------------------
# 7. RUN
# -----------------------------

movie_name = input("Enter movie name: ")

result = chain.invoke({"movie": movie_name})

# -----------------------------
# 8. OUTPUT
# -----------------------------

print("\n================ MOVIE REVIEW ================\n")

print("Title:", result.title)
print("Year:", result.release_year)
print("Genre:", ", ".join(result.genre))
print("Rating:", result.rating)
print("Language:", result.language)
print("Duration:", result.duration)

print("\nDirector:")
print("  Name:", result.director.name)
print("  Nationality:", result.director.nationality)

print("\nCast:")
for actor in result.cast:
    print(f"  - {actor.name} as {actor.role}")

print("\nPlot Summary:", result.plot_summary)
print("\nReview:", result.review)
print("\nRecommendation:", result.recommendation)

print("\nAwards:", result.awards)

print("\nBudget:", result.budget)
print("Box Office:", result.box_office)

print("\nJSON OUTPUT:\n")
print(result.model_dump_json(indent=2))
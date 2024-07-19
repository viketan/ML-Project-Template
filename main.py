from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

# Simulating a prediction function for math score
def predict_math_score(data: InputData) -> int:
    return int((data.reading_score + data.writing_score) / 2)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def submit_data(
    request: Request,
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: int = Form(...),
    writing_score: int = Form(...)
):
    data = InputData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    math_score = predict_math_score(data)
    result_data = data.dict()
    result_data['math_score'] = math_score

    # Pass data to the template
    return templates.TemplateResponse("result.html", {"request": request, **result_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

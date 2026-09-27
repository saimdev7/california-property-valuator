import io
import joblib 
import pandas as pd 
from fastapi import FastAPI, HTTPException, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse
from pydantic import BaseModel, Field 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("../models/price_predictor.joblib")
features = joblib.load("../models/house_feature.joblib")

class HouseFeatures(BaseModel):
    MedInc : float = Field(gt=0, description = "Median Income of Neighbourhood")
    HouseAge : float = Field(gt=0, description = "Average Age of House")
    Averooms : float = Field(gt=0, description = "Average number of rooms")
    Avebedrooms : float = Field(gt=0, description = "Average number of bedrooms")
    Population : float = Field(gt=0, description = "Total Population of Block")
    AveOccupation : float = Field(gt=0, description = "Average people in a house")
    Latitude : float = Field(gt=32, le=42,  description = "Latitude")
    Longitude : float = Field(gt=-125, le=-114 , description = "Longitutde")

@app.get("/")
def home():
    return FileResponse("../static/index.html")

@app.get("/health")
def health():
    return{
        "status": "running",
        "model" : "RandomForrestReggressor",
        "features" : features,
        "avg_error" : "$37,778"
    }

@app.post("/predict")
def predict(application: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": application.MedInc,
            "HouseAge": application.HouseAge,
           "AveRooms": application.Averooms,
           "AveBedrms": application.Avebedrooms,
           "Population": application.Population,
           "AveOccup": application.AveOccupation,
           "Latitude": application.Latitude,
           "Longitude": application.Longitude
           }])
        
        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return{
            "predicted price" : f"${price_usd:,.0f}",
            "findance range" : f"${price_usd - 37778:,.0f} to ${price_usd + 37778:,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"prediction failed : str{e}"   
        )

        
@app.post("/predict-file")
async def predict_file(file:UploadFile=File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code = 400,
            detail = "please upload a .csv file only"
        )   

    contents =  await file.read()

    df = pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
        ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code = 400,
            detail = f"these columns are missing from your file : {missing_columns}"
        )

    if len(df) == 0:
        raise HTTPException(
            status_code = 400,
            detail = "The uploaded file has no data."
        )

    try:
        prediction = model.predict(df[required_columns])
    
        df["predicted_column_usd"] = prediction * 100000
    
        df["predicted_column_usd"] = df["predicted_column_usd"].apply(
            lambda x: f"${x:,.0f}"
        )
    
        output = df.to_csv(index=False)
    
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=prediction.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed: {str(e)}"
        )


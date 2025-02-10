import logging
import pandas as pd
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from src.CrossSellingRS.mainFunctions.CsrsTrain import CrossSellingRSModel
from src.CrossSellingRS.mainFunctions.CsrsPredict import CrossSellingPredict
from src.CrossSellingRS.logging import logging
from src.CrossSellingRS.utils.common import CustomException
# from src.CrossSellingRS.pipeline.stage_06_model_prediction import CrossSellingRSModel
# from src.CrossSellingRS.components.model_trainer import PrmModelArchitecture
# from src.CrossSellingRS.config.configuration import PrmConfigurationManager
# from src.CrossSellingRS.utils.common import load_object
import json
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import numpy as np
import math
import os
import threading
import  schedule
from dotenv import load_dotenv, dotenv_values
import time
import mlflow
import mysql.connector
from pydantic import BaseModel
from typing import List
import requests
import time
import concurrent.futures
import pickle
#Load Environment File
load_dotenv()

#Load Mysql File
mydb = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_DATABASE")
    )

#Server Startup Code
app = FastAPI()


class Service(BaseModel):
    QF_MERCHANT_PRID: int
    QF_MERCHANT_SERVICE_PRID: List[int]
    SERVICE_NAME: List[str]

class RequestData(BaseModel):
    services: List[Service]

#Prm Service Data (Pydantic)
class ServiceData(BaseModel):
    QF_MERCHANT_SERVICE_PRID: Optional[int] = None
    QF_MERCHANT_PERSONNEL_PRID: Optional[int] = None
    SERVICE_NAME: Optional[str] = None
    SERVICE_OFFER_PRICE: Optional[float] = None
    SERVICE_VIEW_COUNT: Optional[int] = None
    category_name: Optional[str] = None
    MERCHANT_PERSONNEL_CREATED_ON_DATE: Optional[str] = None
    STORE_DOCUMENT_VERIFIED_FLAG: Optional[bool] = None
    MERCHANT_NAME: Optional[str] = None
    QF_MERCHANT_PRID: Optional[int] = None
    MERCHANT_WHATSAPP_CONSENT_FLAG: Optional[bool] = None
    MERCHANT_CONTACT_EMAIL: Optional[str] = None
    MERCHANT_CONTACT_PHONE: Optional[str] = None
    serviceBookingCount: Optional[int] = None
    serviceScanario: Optional[str] = None
    serviceRecPrice: Optional[float] = None
    pricePRM: Optional[float] = None

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.on_event("startup")
async def startup_event():


    # Check if the pickle file exists
    if os.path.exists("csrsArtifacts/data_ingestion/ensemblePath.pkl"):
        with open("csrsArtifacts/data_ingestion/ensemblePath.pkl", "rb") as f:
            model = pickle.load(f)
        print("Pickle file loaded")
        # Get the result_df from Redis
        result_df = pd.read_csv("csrsArtifacts/data_transformation/trasnformedData.csv")
        result_df["combined_vectors"] = result_df["combined_vectors"].apply(lambda x: np.array(json.loads(x)))
    else:
        print("Pickle file does not exist, continuing without loading model")

    # # schedule.every(1).minutes.do(train_model)  # Schedule your job
    # schedule.every().wednesday.at("14:04").do(train_model)
    # schedule.every().friday.at("14:04").do(train_model)
    # threading.Thread(target=run_schedule, daemon=True).start()
    
# def register_exception(app: FastAPI):
#     @app.exception_handler(RequestValidationError)
#     async def validation_exception_handler(request: Request, exc: RequestValidationError):

#         exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
#         # or logger.error(f'{exc}')
#         logger.error(request, exc_str)
#         content = {'status_code': 10422, 'message': exc_str, 'data': None}
#         return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

# def train_model():
#     try:
#         model = PrmTrainMlModel(mydb)
#         results = model.train()
#         print("Model trained successfully.")
#     except Exception as e:
#         print(f"Error training model: {e}")

# def run_schedule():
#     while True:
#         schedule.run_pending()
#         time.sleep(1)  # Sleep to prevent high CPU usage

#Api Endpoints
@app.post("/mlModels/prm/predict")
async def predict(finalArray: List[ServiceData]):
    model = PrmPredictModel(finalArray)
    results = model.predict()    
    return results    

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/prm/train")
async def trainModel():
    model = PrmTrainMlModel(mydb)
    results = model.train()
    print("Model Trained successfully")

@app.get("/csrs/train")
async def trainModel():

    global model
    global result_df

    import time
    start = time.time()
    
    model = CrossSellingRSModel(mydb)
    results = model.train()
    print("Model Trained successfully")
    
    end = time.time()
    print("Time taken:", end - start, "seconds")


    # Check if the pickle file exists
    if os.path.exists("csrsArtifacts/data_ingestion/ensemblePath.pkl"):
        with open("csrsArtifacts/data_ingestion/ensemblePath.pkl", "rb") as f:
            model = pickle.load(f)
        print("Pickle file loaded")
        # Get the result_df from Redis
        result_df = pd.read_csv("csrsArtifacts/data_transformation/trasnformedData.csv")
        result_df["combined_vectors"] = result_df["combined_vectors"].apply(lambda x: np.array(json.loads(x)))
    else:
        print("Pickle file does not exist, continuing without loading model")

@app.post("/csrs/cs")
async def crossSellEndPoint(services: List[Service]):
    # Convert each service instance to JSON
    services_json = [service.json() for service in services]

    # Parse each service's JSON string into a dictionary
    parsed_services_json = [json.loads(service) for service in services_json]

    # Pass the parsed JSON data to the CrossSellingPredict class
    CSServices = CrossSellingPredict(mydb, parsed_services_json, model, result_df)
    results = CSServices.predict()
    
    return results

@app.get("/csrs/loadModel")
async def startup_event():
    # Initialize the Redis client
    global model
    global result_df

    # Check if the pickle file exists
    if os.path.exists("csrsArtifacts/data_ingestion/ensemblePath.pkl"):
        with open("csrsArtifacts/data_ingestion/ensemblePath.pkl", "rb") as f:
            model = pickle.load(f)
        print("Pickle file loaded")
        # Get the result_df from Redis
        result_df = pd.read_csv("csrsArtifacts/data_transformation/trasnformedData.csv")
        result_df["combined_vectors"] = result_df["combined_vectors"].apply(lambda x: np.array(json.loads(x)))
    else:
        print("Pickle file does not exist, continuing without loading model")
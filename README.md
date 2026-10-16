# house_rent_prediction
End to end project to predict the house rent price at cities like delhi, mumbai and pune.
change the project id and bucket in the config.
adjust the file path in the modules where required. 
then run 
python -m src.preprocessing
python -m src.modeling
For API, run
uvicorn src.api:app --port 8084



from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = FastAPI()

lr = LinearRegression()
scaler = StandardScaler()

customers = pd.read_csv('USA_Housing.csv')
X = customers.drop(['Price', 'Address'], axis=1)
y = customers['Price']
cols = X.columns
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=101)

lr.fit(X_train, y_train)

pred = lr.predict(X_test)
print(r2_score(y_test,pred))
print(lr.score(X_train, y_train))
mse = mean_squared_error(y_test, pred)
print(mse)

class Item(BaseModel):
    AvgAreaIncome: float
    AvgAreaHouseAge: float
    AvgAreaRooms: float
    AvgAreaBedrooms: float
    AreaPopulation: float

@app.post("/predict_price")
async def predict_price(item: Item):
    features = [item.AvgAreaIncome, item.AvgAreaHouseAge, item.AvgAreaRooms, item.AvgAreaBedrooms, item.AreaPopulation]
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = lr.predict(scaled_features)[0]
    print(prediction)
    return {"predicted_price": prediction}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_item():
    return FileResponse("static/index.html")

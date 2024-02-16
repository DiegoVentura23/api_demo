from fastapi import FastAPI
import uvicorn 
from pydantic import BaseModel


app= FastAPI(title='Demo', version='0.1.0')



@app.get('/demo')
def demo(name :str): 
    return f" Hola {name}"



class InputPredict(BaseModel):
    user_id: str
    n_visits: int


@app.post('/predict')
def predict(input_predict: InputPredict): 
    print(input_predict.user_id)
    print(input_predict.n_visits)
    
    
    return 0.3





if __name__ == '__main__':
    uvicorn.run(app, host= '0.0.0.0', port=8080)
    


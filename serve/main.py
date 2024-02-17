from fastapi import FastAPI
import uvicorn
import joblib
from pydantic_types import InputIris, OutputPredict
import gcsfs


gcs_client= gcsfs.GCSFileSystem()

Target_names= ['setosa', 'versicolor', 'virginica']


model_destination_url= 'gs://ue-model-diego-ventura/iris_classification/model.pkl'
with gcs_client.open(model_destination_url, 'rb') as f:
    pipe = joblib.load(f)

app = FastAPI(title='Iris')


@app.post('/predict', response_model= OutputPredict)
def predict(input_iris: InputIris):
    probs = pipe.predict_proba([[input_iris.petal_lenght, input_iris.petal_width]])

    return{

        'results': [

 
        {name: prob for name, prob in zip(Target_names, p)}
        for p in probs

        ][0]

 
   
    }


if __name__ == '__main__': 
    uvicorn.run (app, host='0.0.0.0', port=8080)

    

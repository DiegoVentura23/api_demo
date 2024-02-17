FROM python:3.10-slim 

WORKDIR /app

COPY serve/main.py /app/
COPY serve/pydantic_types.py /app/
COPY serve/requierements.txt /app/ 

RUN pip install -r requierements.txt

EXPOSE 8080
## EL DOCUMENTO DE LA CARPETA SERVE 

CMD ["python", "main.py"]



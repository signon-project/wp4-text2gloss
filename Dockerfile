FROM python:3.10

LABEL authors="Bram Vanroy"

WORKDIR /app

COPY . /app/
RUN pip install --no-cache-dir --upgrade .

WORKDIR /app/src/text2gloss/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

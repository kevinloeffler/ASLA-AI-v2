FROM python:3.10-bookworm

WORKDIR /app

# install opencv dependencies
RUN apt-get update && apt-get install libgl1 -y

# install pytorch cpu version explicitly
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY /models ./models

COPY /src ./src

WORKDIR /app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

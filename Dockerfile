FROM python:3.10-bookworm

WORKDIR /app

# install pytorch cpu version explicitly
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /models ./models

COPY /src ./src

CMD ["python", "./src/main.py"]

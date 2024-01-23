FROM python:3.10

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt

ENV backend_port $BACKEND_PORT
COPY . /code

CMD ["sh", "-c", "python3 initiate.py"]
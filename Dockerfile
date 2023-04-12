FROM python:3.9-slim-buster

WORKDIR /usr/src/app

COPY poetry.lock pyproject.toml /usr/src/app/

# poetry build dependencies
RUN pip install pip poetry setuptools wheel
RUN poetry config virtualenvs.create false

# opencv libGL.so.1 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN poetry install --no-root

COPY . /usr/src/app

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]

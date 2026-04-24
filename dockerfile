FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn httpx pydantic
EXPOSE 8001 8002 8003 8004
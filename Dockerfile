# Base image com Python 3.11 slim para otimização de tamanho e segurança
FROM python:3.11-slim

# Metadados da imagem
LABEL maintainer="BH Transit AI Research Group"
LABEL description="BH Transit AI - Mobilidade Urbana e Inteligência Espaço-Temporal"

# Define variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema necessárias para compilação e GIS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho no container
WORKDIR /app

# Copia e instala dependências Python com cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo o código-fonte e dados
COPY . .

# Expõe as portas do Streamlit (8504) e FastAPI (8000)
EXPOSE 8504 8000

# Comando padrão: inicia o Streamlit
CMD ["streamlit", "run", "src/app_mapa.py", "--server.port=8504", "--server.address=0.0.0.0"]

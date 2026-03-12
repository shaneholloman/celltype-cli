# MSA Search — Local MMseqs2 homology search
# Uses local ColabFold databases (UniRef30 + EnvDB)
# Hardware: CPU only, no GPU required
# Databases must be mounted at /vol/colabfold_db

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates \
    && wget -qO- https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | tar xz \
    && cp mmseqs/bin/mmseqs /usr/local/bin/ \
    && rm -rf mmseqs \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV COLABFOLD_DB=/vol/colabfold_db

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]

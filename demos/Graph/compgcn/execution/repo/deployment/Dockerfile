FROM python:3.11

RUN useradd -m -u 1000 user && python -m pip install --upgrade pip
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
COPY --chown=user ./deployment/install_deps.py deployment/install_deps.py
RUN python deployment/install_deps.py

COPY --chown=user . /app

FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory for the application
WORKDIR /app

# Copy the uv package manager from its latest image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Add the virtual environment's bin directory to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency and environment configuration files
COPY "dependency/pyproject.toml" "dependency/uv.lock" "dependency/.python-version" ./

# Install dependencies exactly as pinned in uv.lock
RUN uv sync --locked

COPY "starter.py" ./

# Run starter.py
CMD ["uv", "run", "python", "starter.py", "2023", "5"]
# AI Service Platform

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd ai_service_platform

# Copy environment file
cp .env.example .env

# Start with Docker Compose
docker-compose up -d

# Or run locally
pip install -e .
uvicorn src.main:app --reload
```

## Documentation

See [docs/README.md](docs/README.md) for full documentation.

## License

MIT

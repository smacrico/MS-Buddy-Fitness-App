---
title: "APEX Run Analysis API Reference"
summary: "Complete API documentation for the APEX Run Analysis service endpoints"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "src/api/*"
  - "src/models/*"
---

# API Reference

**Who should read this**: Developers integrating with APEX Run Analysis
**Estimated reading time**: 20 minutes

## API Overview

The APEX Run Analysis API provides REST endpoints for analyzing running performance data.

### Base URL
```
https://api.ms-buddy-fitness.com/apex/v1
```

### Authentication
All endpoints require Bearer token authentication:
```http
Authorization: Bearer <access_token>
```

## Endpoints

### Analyze Run Data
Analyzes a single running session.

```http
POST /analyze/run
```

#### Request Body
```json
{
  "userId": "string",
  "runData": {
    "startTime": "2025-10-22T08:00:00Z",
    "duration": 3600,
    "distance": 10000,
    "gpsPoints": [
      {
        "latitude": 47.6062,
        "longitude": -122.3321,
        "timestamp": "2025-10-22T08:00:00Z",
        "elevation": 100
      }
    ],
    "heartRateData": [
      {
        "bpm": 140,
        "timestamp": "2025-10-22T08:00:00Z"
      }
    ]
  }
}
```

#### Response
```json
{
  "analysisId": "string",
  "metrics": {
    "averagePace": 360,
    "totalDistance": 10000,
    "elevationGain": 150,
    "estimatedVO2Max": 45.5
  },
  "insights": [
    {
      "type": "PACE_VARIATION",
      "severity": "INFO",
      "message": "Consistent pacing throughout the run"
    }
  ],
  "recommendations": [
    {
      "type": "TRAINING",
      "message": "Consider increasing duration by 10% next session"
    }
  ]
}
```

### Get User Performance Trends
Retrieves running performance trends over time.

```http
GET /trends/{userId}
```

#### Query Parameters
- `startDate`: ISO8601 date string
- `endDate`: ISO8601 date string
- `metric`: Metric type (pace|distance|vo2max)

## Error Responses

All errors follow this format:
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

### Common Error Codes
- `400`: Invalid request
- `401`: Unauthorized
- `404`: Resource not found
- `429`: Rate limit exceeded
- `500`: Internal server error
```

Now, let's create the Developer Guide:

````markdown
// filepath: docs/wiki/06_Developer_Guide.md
---
title: "APEX Run Analysis Developer Guide"
summary: "Comprehensive guide for developers working on the APEX Run Analysis system"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "README.md"
  - "docs/*"
  - "src/**/*"
---

# Developer Guide

**Who should read this**: Developers contributing to APEX Run Analysis
**Estimated reading time**: 25 minutes

## Development Environment Setup

### Required Tools
- Visual Studio Code
- Python 3.9+
- Git
- Docker Desktop (optional)

### VS Code Extensions
- Python
- Pylance
- Python Test Explorer
- Docker

### Environment Setup Steps

1. **Clone and Configure Repository**
```bash
git clone https://github.com/MS-Buddy-Fitness-App/APEX-RunAnalysis.git
cd APEX-RunAnalysis
```

2. **Create Python Environment**
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Configure VS Code**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.testing.pytestEnabled": true,
    "python.formatting.provider": "black"
}
```

## Project Structure

```
APEX-RunAnalysis/
├── src/
│   ├── analysis/       # Core analysis logic
│   ├── api/           # FastAPI routes and handlers
│   ├── models/        # Data models and ML models
│   └── utils/         # Shared utilities
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
└── scripts/
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_analysis.py

# Run with coverage
pytest --cov=src tests/
```

### 3. Code Quality Checks
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Debugging

### VS Code Launch Configuration
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "src.main:app",
                "--reload",
                "--port",
                "8000"
            ]
        }
    ]
}
```

### Debug Tips
1. Use `breakpoint()` in code
2. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Common Development Tasks

### Adding New Analysis Feature
1. Create new module in `src/analysis/`
2. Add unit tests in `tests/unit/`
3. Update API endpoints if needed
4. Update documentation

### Adding ML Model
1. Place model file in `src/models/`
2. Create model wrapper class
3. Add model tests
4. Update model registry

## Performance Optimization

### Profiling Code
```python
import cProfile
prof = cProfile.Profile()
prof.enable()
# Your code here
prof.disable()
prof.print_stats(sort='cumtime')
```

### Memory Usage
- Use generators for large datasets
- Implement pagination in APIs
- Profile memory with `memory_profiler`

## Best Practices

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Document public APIs

2. **Testing**
   - Write tests first (TDD)
   - Mock external dependencies
   - Use fixtures for common test data

3. **Git Commits**
   - Use conventional commits
   - Keep commits atomic
   - Write meaningful messages

4. **Documentation**
   - Update docs with code changes
   - Include examples
   - Document assumptions
```

Would you like me to continue with generating the remaining documentation files?// filepath: docs/wiki/05_API_Reference.md
---
title: "APEX Run Analysis API Reference"
summary: "Complete API documentation for the APEX Run Analysis service endpoints"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "src/api/*"
  - "src/models/*"
---

# API Reference

**Who should read this**: Developers integrating with APEX Run Analysis
**Estimated reading time**: 20 minutes

## API Overview

The APEX Run Analysis API provides REST endpoints for analyzing running performance data.

### Base URL
```
https://api.ms-buddy-fitness.com/apex/v1
```

### Authentication
All endpoints require Bearer token authentication:
```http
Authorization: Bearer <access_token>
```

## Endpoints

### Analyze Run Data
Analyzes a single running session.

```http
POST /analyze/run
```

#### Request Body
```json
{
  "userId": "string",
  "runData": {
    "startTime": "2025-10-22T08:00:00Z",
    "duration": 3600,
    "distance": 10000,
    "gpsPoints": [
      {
        "latitude": 47.6062,
        "longitude": -122.3321,
        "timestamp": "2025-10-22T08:00:00Z",
        "elevation": 100
      }
    ],
    "heartRateData": [
      {
        "bpm": 140,
        "timestamp": "2025-10-22T08:00:00Z"
      }
    ]
  }
}
```

#### Response
```json
{
  "analysisId": "string",
  "metrics": {
    "averagePace": 360,
    "totalDistance": 10000,
    "elevationGain": 150,
    "estimatedVO2Max": 45.5
  },
  "insights": [
    {
      "type": "PACE_VARIATION",
      "severity": "INFO",
      "message": "Consistent pacing throughout the run"
    }
  ],
  "recommendations": [
    {
      "type": "TRAINING",
      "message": "Consider increasing duration by 10% next session"
    }
  ]
}
```

### Get User Performance Trends
Retrieves running performance trends over time.

```http
GET /trends/{userId}
```

#### Query Parameters
- `startDate`: ISO8601 date string
- `endDate`: ISO8601 date string
- `metric`: Metric type (pace|distance|vo2max)

## Error Responses

All errors follow this format:
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

### Common Error Codes
- `400`: Invalid request
- `401`: Unauthorized
- `404`: Resource not found
- `429`: Rate limit exceeded
- `500`: Internal server error
```

Now, let's create the Developer Guide:

````markdown
// filepath: docs/wiki/06_Developer_Guide.md
---
title: "APEX Run Analysis Developer Guide"
summary: "Comprehensive guide for developers working on the APEX Run Analysis system"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "README.md"
  - "docs/*"
  - "src/**/*"
---

# Developer Guide

**Who should read this**: Developers contributing to APEX Run Analysis
**Estimated reading time**: 25 minutes

## Development Environment Setup

### Required Tools
- Visual Studio Code
- Python 3.9+
- Git
- Docker Desktop (optional)

### VS Code Extensions
- Python
- Pylance
- Python Test Explorer
- Docker

### Environment Setup Steps

1. **Clone and Configure Repository**
```bash
git clone https://github.com/MS-Buddy-Fitness-App/APEX-RunAnalysis.git
cd APEX-RunAnalysis
```

2. **Create Python Environment**
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Configure VS Code**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.testing.pytestEnabled": true,
    "python.formatting.provider": "black"
}
```

## Project Structure

```
APEX-RunAnalysis/
├── src/
│   ├── analysis/       # Core analysis logic
│   ├── api/           # FastAPI routes and handlers
│   ├── models/        # Data models and ML models
│   └── utils/         # Shared utilities
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
└── scripts/
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_analysis.py

# Run with coverage
pytest --cov=src tests/
```

### 3. Code Quality Checks
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Debugging

### VS Code Launch Configuration
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "src.main:app",
                "--reload",
                "--port",
                "8000"
            ]
        }
    ]
}
```

### Debug Tips
1. Use `breakpoint()` in code
2. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Common Development Tasks

### Adding New Analysis Feature
1. Create new module in `src/analysis/`
2. Add unit tests in `tests/unit/`
3. Update API endpoints if needed
4. Update documentation

### Adding ML Model
1. Place model file in `src/models/`
2. Create model wrapper class
3. Add model tests
4. Update model registry

## Performance Optimization

### Profiling Code
```python
import cProfile
prof = cProfile.Profile()
prof.enable()
# Your code here
prof.disable()
prof.print_stats(sort='cumtime')
```

### Memory Usage
- Use generators for large datasets
- Implement pagination in APIs
- Profile memory with `memory_profiler`

## Best Practices

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Document public APIs

2. **Testing**
   - Write tests first (TDD)
   - Mock external dependencies
   - Use fixtures for common test data

3. **Git Commits**
   - Use conventional commits
   - Keep commits atomic
   - Write meaningful messages

4. **Documentation**
   - Update docs with code changes
   - Include examples
   - Document assumptions
```

Would you like me to continue with generating the remaining documentation files?
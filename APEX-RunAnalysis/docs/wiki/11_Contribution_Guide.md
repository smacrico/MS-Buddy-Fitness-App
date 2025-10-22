---
title: "APEX Run Analysis Contribution Guide"
summary: "Guidelines for contributing to the project"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "CONTRIBUTING.md"
  - ".github/*"
---

# Contribution Guide

**Who should read this**: Developers contributing to the project
**Estimated reading time**: 12 minutes

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Code Style

### Python Style Guide
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters
- Use Black for formatting

### Commit Messages
```
type(scope): subject

body

footer
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Formatting changes
- refactor: Code restructuring
- test: Adding/modifying tests
- chore: Maintenance tasks

## Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review from maintainers

## Pull Request Template

```markdown
// filepath: .github/pull_request_template.md
## Description
[Describe the changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests passing
- [ ] PR is appropriately scoped
```
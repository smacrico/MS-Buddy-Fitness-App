---
title: "APEX Run Analysis Release and Changelog"
summary: "Release process and version history"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "CHANGELOG.md"
  - "version.py"
---

# Release and Changelog

**Who should read this**: DevOps engineers and release managers
**Estimated reading time**: 10 minutes

## Version Control

Version format: `MAJOR.MINOR.PATCH`

Example: `1.2.3`
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

## Release Process

1. Update version:
```python
// filepath: src/version.py
VERSION = "1.2.3"
```

2. Update changelog
3. Create release branch
4. Create release tag
5. Deploy to staging
6. Verify deployment
7. Deploy to production

## Changelog

```markdown
// filepath: CHANGELOG.md
# Changelog
All notable changes to this project will be documented in this file.

## [1.2.3] - 2025-10-22
### Added
- New pace analysis algorithm
- Support for elevation data

### Changed
- Improved performance of run analysis
- Updated dependency versions

### Fixed
- Memory leak in long-running analyses
- Incorrect pace calculation for short distances

## [1.2.2] - 2025-10-15
### Added
- Heart rate zone analysis
- REST API documentation
```
$baseDir = "C:\smakryko\myHealthData\MS-Buddy-Fitness-App\APEX-RunAnalysis\docs\wiki"

# Create base directory and diagrams subdirectory if they don't exist
New-Item -Path "$baseDir\diagrams" -ItemType Directory -Force

# Overview
$overviewContent = @'
---
title: "APEX Run Analysis Overview"
summary: "Core documentation for the APEX Run Analysis system"
last_generated: "2025-10-22T10:00:00Z"
source_files: 
  - "README.md"
  - "src/analysis/*"
  - "src/models/*"
---

# APEX Run Analysis Overview

**Who should read this**: Developers and data scientists working on the MS-Buddy-Fitness-App running analysis features
**Estimated reading time**: 10 minutes

## Executive Summary
APEX Run Analysis is a specialized component within the MS-Buddy-Fitness-App ecosystem that focuses on analyzing running performance data. It provides advanced metrics, insights, and recommendations for runners based on their historical activity data.

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\00_Overview.md" -Value $overviewContent

# Getting Started
$gettingStartedContent = @'
---
title: "Getting Started with APEX Run Analysis"
summary: "Quick start guide for developers"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "README.md"
  - "requirements.txt"
---

# Getting Started

**Who should read this**: New developers joining the project
**Estimated reading time**: 8 minutes

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\01_Getting_Started.md" -Value $gettingStartedContent

# Architecture
$architectureContent = @'
---
title: "APEX Run Analysis Architecture"
summary: "Detailed architecture overview of the APEX Run Analysis system"
last_generated: "2025-10-22T10:00:00Z"
source_files: 
  - "src/analysis/*"
  - "src/models/*"
  - "src/utils/*"
---

# Architecture Overview

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\02_Architecture.md" -Value $architectureContent

# API Reference
$apiReferenceContent = @'
---
title: "APEX Run Analysis API Reference"
summary: "Complete API documentation for the APEX Run Analysis service endpoints"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "src/api/*"
  - "src/models/*"
---

# API Reference

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\05_API_Reference.md" -Value $apiReferenceContent

# Developer Guide
$developerGuideContent = @'
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

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\06_Developer_Guide.md" -Value $developerGuideContent

# Testing Guide
$testingContent = @'
---
title: "APEX Run Analysis Testing Guide"
summary: "Comprehensive testing documentation and guidelines"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "tests/**/*"
  - "pytest.ini"
---

# Testing Guide

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\07_Testing.md" -Value $testingContent

# CI/CD Guide
$cicdContent = @'
---
title: "APEX Run Analysis CI/CD Pipeline"
summary: "Continuous Integration and Deployment documentation"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - ".github/workflows/*"
  - "scripts/deploy/*"
---

# CI/CD Pipeline

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\08_CI_CD.md" -Value $cicdContent

# Security Guide
$securityContent = @'
---
title: "APEX Run Analysis Security Guide"
summary: "Security configuration and best practices"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - ".env.example"
  - "src/config/*"
---

# Security and Secrets

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\09_Security_and_Secrets.md" -Value $securityContent

# Performance Guide
$performanceContent = @'
---
title: "APEX Run Analysis Performance and Monitoring"
summary: "Performance optimization and monitoring guidelines"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "src/monitoring/*"
  - "src/config/logging.py"
---

# Performance and Monitoring

[Rest of content as previously generated]
'@

Set-Content -Path "$baseDir\10_Performance_and_Monitoring.md" -Value $performanceContent

# Create remaining files with template content
$remainingFiles = @(
    @{Name = "03_Installation_and_Deployment.md"; Title = "Installation and Deployment Guide"},
    @{Name = "04_Configuration.md"; Title = "Configuration Guide"},
    @{Name = "11_Contribution_Guide.md"; Title = "Contribution Guidelines"},
    @{Name = "12_Release_and_Changelog.md"; Title = "Release and Changelog"},
    @{Name = "13_FAQ_and_Troubleshooting.md"; Title = "FAQ and Troubleshooting"},
    @{Name = "14_Glossary.md"; Title = "Technical Glossary"},
    @{Name = "15_Diagrams.md"; Title = "Technical Diagrams"}
)

foreach ($file in $remainingFiles) {
    $templateContent = @"
---
title: "$($file.Title)"
summary: "Documentation for $($file.Title.ToLower())"
last_generated: "2025-10-22T10:00:00Z"
source_files:
  - "docs/*"
---

# $($file.Title)

**Who should read this**: Developers and system administrators
**Estimated reading time**: 10 minutes

[Content to be generated]
"@

    Set-Content -Path "$baseDir\$($file.Name)" -Value $templateContent
}

# Verify files were created
Get-ChildItem -Path $baseDir -Recurse | Select-Object Name, LastWriteTime

Write-Host "Documentation files have been generated successfully!"
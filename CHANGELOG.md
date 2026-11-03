# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-03
### Added
- Initial release of Tabular-Gauge.
- Configuration-driven `ModelFactory` for dynamic model instantiation.
- Support for both OpenML and custom CSV datasets.
- Comprehensive evaluation suite covering performance, calibration, and efficiency.
- Statistical testing layer (Friedman, Nemenyi, Wilcoxon) with Holm-Bonferroni correction.
- Automated report generation for summary tables and figures.
- Lazy loading of model dependencies via optional install groups.
- Professional project infrastructure (pyproject.toml, CONTRIBUTING, USER_GUIDE).

### Changed
- Refactored package namespace to `tabular_gauge`.
- Implemented centralized metric-direction registry to ensure correct rankings for loss metrics.
- Improved README and User Guide for better onboarding.

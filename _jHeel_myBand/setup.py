from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_desc = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="longevity-metrics",
    version="0.1.0",
    author="MS Buddy Fitness Team",
    description="Compute health metrics from wearable device data",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'jupyter>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'longevity=longevity.cli:main',
        ],
    },
)

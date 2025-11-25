from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='longevity-metrics',
    version='1.0.0',
    author='jHeel MyBand Team',
    author_email='',
    description='Health and longevity metrics computation from wearable device data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smacrico/MS-Buddy-Fitness-App',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'openpyxl>=3.0.0',
        'pyarrow>=5.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
    },
    entry_points={
        'console_scripts': [
            'longevity=longevity.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# setup.py
# File path: setup.py

"""
Setup script for EEE Pipeline CLI installation.

Install in development mode:
    pip install -e .

Install in production mode:
    pip install .

After installation, the CLI is available as:
    eee-cli <command>
"""

from setuptools import setup, find_packages

setup(
    name="eee-pipeline",
    version="1.0.0",
    description="Event & Entity Extraction (EEE) Pipeline - CLI and API",
    author="Your Organization",
    author_email="admin@example.com",
    packages=find_packages(),
    install_requires=[
        "click==8.1.7",
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "pydantic==2.7.1",
        "httpx==0.27.0",
        "celery==5.3.6",
        "redis==5.0.1",
        "dask==2024.5.1",
        "distributed==2024.5.1",
        "python-json-logger==2.0.7",
        "PyYAML==6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "eee-cli=src.cli.main:cli",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


# setup.py
# File path: setup.py

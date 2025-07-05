from setuptools import setup, find_packages

setup(
    name="backtester",
    version="1.0.0",
    description="Binance arbitrage strategy backtester with automatic data download",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.21.0",
        "pyarrow>=10.0.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "asyncio",
        "logging",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "backtester=backtester.cli:main",
        ],
    },
) 
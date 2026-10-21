from setuptools import setup, find_packages

setup(
    name='MirrorGPT',
    version='2.0.0',
    description='Autonomous AI Digital Twin & Personal Alignment Engine',
    author='MirrorGPT Team',
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.110.0',
        'uvicorn>=0.28.0',
        'pydantic>=2.0.0',
        'python-multipart>=0.0.9',
        'pypdf>=4.0.0',
        'sentence-transformers>=2.5.0',
        'numpy>=1.24.0',
        'requests>=2.31.0',
        'python-dotenv>=1.0.0'
    ],
    python_requires='>=3.9',
)

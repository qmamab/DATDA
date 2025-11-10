from setuptools import setup, find_packages

setup(
    name="datda",  # The package name
    version="0.1.0",
    author="Qamar Muneer Akbar",
    author_email="qamar@ftiuae.com",
    description="DATDA — Defense Against The Dark Arts: Inference-time image purifier and robust CNN defense pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/qmamab/DATDA",  
    packages=find_packages(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.17.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "gradio>=3.50.0",
        "scipy>=1.11.0",
        "timm>=0.9.0"  
    ],
    include_package_data=True, 
)

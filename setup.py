from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hmm-dmd-wifi-localization",
    version="0.1.0",
    author="Oluwaseyi Paul Babalola",
    description="WiFi Fingerprinting Indoor Localization using HMM and DMD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/babalolaseyip/HMM-DMD-WiFi-Localization",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0", "black", "flake8"],
    },
)

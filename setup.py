from pathlib import Path
from setuptools import setup, find_namespace_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ica-aroma-py",
    version="0.1.4",
    description="ICA-AROMA packaged for Python import usage.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LICE - Commissione Neuroimmagini",
    author_email="dev@lice.it",
    license="Apache-2.0",
    url="https://github.com/LICE-dev/ICA-AROMA-PY",
    project_urls={
        "Homepage": "https://github.com/LICE-dev/ICA-AROMA-PY",
        "Source": "https://github.com/LICE-dev/ICA-AROMA-PY",
        "Bug Tracker": "https://github.com/LICE-dev/ICA-AROMA-PY/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Intended Audience :: Science/Research",
    ],
    keywords=["fMRI", "ICA-AROMA", "neuroimaging", "motion artifacts", "nipype"],
    packages=find_namespace_packages(include=["ica_aroma_py*"]),
    include_package_data=True,
    package_data={
        "ica_aroma_py": ["resources/*.nii.gz"],
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.2.4",
        "nibabel>=5.3.0,<6",
    ],
    extras_require={
        "nipype": [
            "nipype>=1.12.0",
        ],
        "plots": [
            "pandas",
            "matplotlib>=3.10.1",
            "seaborn>=0.13.2,<0.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "ica-aroma-py=ica_aroma_py.services.cli:main",
        ]
    },
)

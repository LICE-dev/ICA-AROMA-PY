from setuptools import setup, find_packages

setup(
    name="ica-aroma-py",
    version="0.1.0",
    description="ICA-AROMA packaged for Python import usage.",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ica_aroma_py": ["resources/*.nii.gz"],
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.2.4",
        # Se per ora tieni ancora future/past in ICA_AROMA_functions.py, aggiungi:
        # "future",
    ],
    extras_require={
        "plots": [
            "pandas",
            "matplotlib==3.10.1",
            "seaborn>=0.13.2,<0.14",
        ],
    },
)

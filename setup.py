from setuptools import setup, find_packages

setup(
    name="ica-aroma-py",
    version="0.1.1",
    description="ICA-AROMA packaged for Python import usage.",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ica_aroma_py": ["resources/*.nii.gz"],
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.2.4",
        "nibabel>=5.3.0,<6",
    ],
    extras_require={
        "nipype": [
            "nipype==1.10.0",
        ],
        "plots": [
            "pandas",
            "matplotlib==3.10.1",
            "seaborn>=0.13.2,<0.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "ica-aroma-py=ica_aroma_py.services.cli:main",
        ]
    },
)

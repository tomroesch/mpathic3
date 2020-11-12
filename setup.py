import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mpathic3",
    version="0.0.1",
    author="Tom Roeschinger",
    author_email="troeschi@caltech.edu",
    description="This repository contains a minimal version of mpathic working with pymc3.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/tomroesch/Reg-Seq2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

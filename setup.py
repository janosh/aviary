from setuptools import find_namespace_packages, setup

setup(
    name="roost",
    version="0.0.1",
    author="Rhys Goodall",
    author_email="reag2@cam.ac.uk",
    description="Representation Learning from Stoichiometry",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/comprhys/roost",
    packages=find_namespace_packages(include=["roost*"]),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

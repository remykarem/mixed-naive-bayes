import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mixed-naive-bayes",
    version="0.0.2",
    author="Raimi bin Karim",
    author_email="raimi.bkarim@gmail.com",
    description="Categorical and Gaussian Naive Bayes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/remykarem/mixed-naive-bayes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.16.1', 'scikit-learn>=0.20.2']
)

import setuptools
import io
import os

NAME = "ixai"
DESCRIPTION = "Explainable Artificial Intelligence for dynamic and Incremental models."
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/mmschlk/iXAI"
EMAIL = "maximilian.muschalik@ifi.lmu.de"
AUTHOR = "Maximilian Muschalik"
REQUIRES_PYTHON = ">=3.8.0"

wrkdir = os.path.abspath(os.path.dirname(__file__))
version: dict = {}
with open(os.path.join(wrkdir, NAME, "__version__.py")) as f:
    exec(f.read(), version)

with io.open(os.path.join(wrkdir, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

base_packages = [
    "river",
    "pandas",
    "numpy",
    "tqdm"
]

plot_packages = [
    "matplotlib"
]

dev_packages = [
    "pytest",
    "scikit-learn",
    "flake8",
    "torch",
    "openml"
]

doc_packages = [
    "sphinx",
    #"myst_nb",
    "nbsphinx",  # for rendering jupyter notebooks
    "pandoc",  # for rendering jupyter notebooks
    "furo",  # theme of the docs
    "sphinx-copybutton",  # easier copy-pasting of code snippets from docs
    "myst-parser"  # parse md and rst files
]

setuptools.setup(
    name=NAME,
    version=version["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls={
        "Tracker": "https://github.com/mmschlk/iXAI/issues?q=is%3Aissue+label%3Abug",
        "Source": "https://github.com/mmschlk/iXAI"
    },
    packages=setuptools.find_packages(exclude=('tests', 'examples', 'docs')),
    install_requires=base_packages,
    extras_require={
        "plot": base_packages + plot_packages,
        "dev": base_packages + plot_packages + dev_packages,
        "docs": base_packages + plot_packages + dev_packages + doc_packages,
    },
    include_package_data=True,
    license="MIT",
    license_files=("LICENSE",),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],
    keywords=["python", "machine learning", "online learning", "xai"],
    zip_safe=True
)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from sphinx.builders.html import StandaloneHTMLBuilder
import os
import sys
import increment_explain


sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IncrementExplain'
copyright = '2022, Maximilian Muschalik, Rohit Jagtani'
author = 'Maximilian Muschalik, Rohit Jagtani'
release = increment_explain.__version__
version = increment_explain.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinx_toolbox.more_autodoc.autoprotocol",
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None)
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_title = f"{project} {version} Documentation"
html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "auto"}
html_theme_options = {
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/mmschlk/IncrementExplain.git", "icon": "fab fa-github-square"},
    ],
    "collapse_navigation": True,
    "navigation_depth": 3,
    "show_toc_level": 1,
    "footer_items": ["copyright"],
    "navbar_align": "content",
    "show_prev_next": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# set priority when building html
StandaloneHTMLBuilder.supported_image_types = ["image/svg+xml", "image/gif", "image/png", "image/jpeg"]
# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

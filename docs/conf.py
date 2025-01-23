# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from squlearn import __version__ as squlearn_version

# -- Project information -----------------------------------------------------

project = "sQUlearn"
copyright = "2023, Fraunhofer IPA"
author = "Fraunhofer IPA"

# The full version, including alpha/beta/rc tags
release = squlearn_version
version = squlearn_version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.spelling",
    "sphinx_sitemap",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "jupyter_sphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_default_options = {"members": True, "inherited-members": True}


# Skip property members --> They should be defined in Attributes
def skip_property_member(app, what, name, obj, skip, options):
    if isinstance(obj, property):
        return True


def setup(app):
    app.connect("autodoc-skip-member", skip_property_member)


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "**.ipynb_checkpoints"]

# generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
}
html_css_files = [
    "css/custom.css",
]

latex_engine = "xelatex"
latex_elements = {
    "preamble": r"""
    \usepackage{braket}
    """,
}

# intersphinx
intersphinx_mapping = {
    "pennylane": ("https://docs.pennylane.ai/en/stable/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
    "qiskit-aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "qiskit-ibm-runtime": ("https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/1.4/", None),
}

suppress_warnings = ["myst.header", "config.cache"]

# base URL for sphinx_sitemap
html_baseurl = "https://squlearn.github.io/"
sitemap_url_scheme = "{link}"

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Points to the project root

project = 'MOAAP'
copyright = '2025, Andreas F. Prein, Raphael Graf'
author = 'Andreas F. Prein, Raphael Graf'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Essential for specific docstring format!
    'sphinx.ext.viewcode',  # adds links to source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_custom_sections = [
    ('Precipitation Parameters', 'params_style'),
    ('Moisture Stream Parameters', 'params_style'),
    ('Cyclone & Anticyclone Parameters', 'params_style'),
    ('Frontal Zone Parameters', 'params_style'),
    ('Cloud Tracking Parameters', 'params_style'),
    ('Atmospheric River Parameters', 'params_style'),
    ('Tropical Cyclone Parameters', 'params_style'),
    ('MCS Parameters', 'params_style'),
    ('Jet Stream & Wave Parameters', 'params_style'),
    ('500 hPa Cyclone Parameters', 'params_style'),
    ('Equatorial Wave Parameters', 'params_style'),
]
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
    ('Precipitation objects', 'params_style'),
    ('Moisture streams', 'params_style'),
    ('Cyclones & anticyclones', 'params_style'),
    ('Frontal zones', 'params_style'),
    ('Cloud tracking', 'params_style'),
    ('Atmospheric rivers (AR)', 'params_style'),
    ('Tropical cyclone detection', 'params_style'),
    ('Mesoscale convective systems (MCS)', 'params_style'),
    ('Jet streams & tropical waves', 'params_style'),
    ('500 hPa cyclones/anticyclones', 'params_style'),
    ('Equatorial waves', 'params_style'),
]
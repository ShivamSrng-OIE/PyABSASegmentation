# -*- coding: utf-8 -*-
# file: setup.py
from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")

def read_version():
    """Parse __version__ from pyabsa/__init__.py without importing the package."""
    init_py = (ROOT / "pyabsa" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_py, re.M)
    if not m:
        raise RuntimeError("Cannot find __version__ in pyabsa/__init__.py")
    return m.group(1)

def load_requirements(fname="requirements.txt"):
    """
    Load requirements from requirements.txt, keeping PEP 508 direct URLs and git+ entries.
    Filters out:
      - comments/blank lines
      - editable installs (-e ...)
      - options (lines starting with -- or -c/-r)
      - self-dependency (pyabsa==... or pyabsa @ ...)
      - build-system tools that shouldn't be runtime deps (setuptools, wheel)
    """
    req_path = ROOT / fname
    if not req_path.exists():
        return []
    reqs = []
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e ") or line.startswith("--") or line.startswith("-c ") or line.startswith("-r "):
            continue
        # drop self-dependency if present
        lower = line.lower().replace(" ", "")
        if lower.startswith("pyabsa==") or lower.startswith("pyabsa@"):
            continue
        # drop build tools from runtime deps
        if lower.startswith("setuptools") or lower.startswith("wheel"):
            continue
        reqs.append(line)
    return reqs

VERSION = read_version()
INSTALL_REQUIRES = load_requirements("requirements.txt")

extras = {
    "docs": [
        "recommonmark",
        "nbsphinx",
        "sphinx-autobuild",
        "sphinx-rtd-theme",
        "sphinx-markdown-tables",
        "sphinx-copybutton",
        "piccolo_theme",
    ],
    "test": ["docformatter", "isort", "flake8", "pytest", "pytest-xdist"],
    "deploy": ["twine", "wheel", "setuptools", "gradio"],
}
extras["dev"] = extras["docs"] + extras["test"] + extras["deploy"]

setup(
    name="pyabsa",
    version=VERSION,
    description=(
        "State-of-the-art models for Aspect Term Extraction (ATE), "
        "Aspect Polarity Classification (APC), and Text Classification (TC)."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yangheng95/PyABSA",
    author="Yang, Heng",
    author_email="hy345@exeter.ac.uk",
    python_requires=">=3.10,<3.13",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    install_requires=INSTALL_REQUIRES,
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
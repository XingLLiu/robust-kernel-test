"""
run "pip install -e ." to setup
"""
from setuptools import setup, find_packages

# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()
    

setup(
    name="robust-kernel-test",
    
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="1.0.0",

    # Choose your license
    license="MIT",

    # What does your project relate to?
    keywords="hypothesis-test kernel-methods robustnesss goodness-of-fit",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(include=["rksd"]),

    # See https://www.python.org/dev/peps/pep-0440/#version-specifiers
    python_requires=">= 3.9",

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=parse_requirements('requirements.txt'),
    # install_requires=[
    #     "numpy==1.26.4",
    #     "jax[cpu]==0.4.30",
    #     "pandas==2.2.2",
    #     "scipy==1.13.1",
    #     "matplotlib==3.9.0",
    #     "notebook>=7.0",
    #     "ipywidgets==8.1.3",
    #     "scikit-learn==1.5.0",
    #     "tqdm==4.66.4",
    #     "seaborn==0.13.2",
    #     "jaxopt",
    #     "rdata",
    #     # "git+https://github.com/wittawatj/kernel-gof.git@master#egg=kernel-gof" # for Gaussian-Bernoulli RBM example
    #     "kernel-gof"
    # ],
    # dependency_links=[
    #     'git+https://github.com/wittawatj/kernel-gof.git@master#egg=kernel-gof'
    # ],
)

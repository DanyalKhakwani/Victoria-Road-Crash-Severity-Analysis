from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    if '-e .' in requirements:
        requirements.remove('-e .')
    return requirements

setup(
    name='crash_severity_analysis',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements()
)
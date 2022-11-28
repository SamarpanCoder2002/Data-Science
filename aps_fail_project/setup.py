from setuptools import find_packages, setup
from typing import List

REQUIREMENT_FILE_NAME='requirements.txt'
HYPHEN_E_DOT="-e ." # Used for trigger the setup.py file.... 
# It mentioned in requirements.txt file
# "-e ." used to denote that that project is going to use as a python lib

# Artifact in Machine Learning:

# An artifact is a machine learning term that is used to describe the output created by the training process. Output could be a fully trained model, a model checkpoint, or a file created during the training process


def get_requirements() -> List[str]:
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n","")  for requirement_name in requirement_list]

    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list


setup(
    name='sensor',
    version='0.0.1',
    author='ineuron',
    author_email='avnish@ineuron.ai',
    packages=find_packages(),
    install_requires=get_requirements()
)



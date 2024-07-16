from setuptools import find_packages, setup
from typing import List

DOT_E = "- .e"
def get_requirements(filepath:str)-> List:
    req_list = []
    with open(filepath) as file:
        reqs =file.readline()
        req_list = [req.replace("\n","") for req in reqs]
    if DOT_E in reqs:
        reqs(DOT_E)
    return       

setup(name="ML Project Template",
      version="0.0.1",
      author="Viketan",
      author_email="viketanrevankar108@gmail.com",
      packages= find_packages(),
      install_requires = get_requirements("requirement.txt")
)
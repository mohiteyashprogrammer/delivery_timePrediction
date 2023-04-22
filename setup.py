from setuptools import setup,find_packages
from typing import List

HYPHAN_E_DOT = "-e ."


def get_requirements(file_path:str)->List[str]:
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","") for i in requirements]

        if HYPHAN_E_DOT in requirements:
            requirements.remove(HYPHAN_E_DOT)

    return requirements



setup(
    name="DeliveryTimePradiction",
    version="0.1",
    author="yash mohite",
    author_email="mohite.yassh@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)


















['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
       'Item_MRP', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Outlet_Age']
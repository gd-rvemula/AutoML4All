from setuptools import setup

setup(
   name='AutoML4All',
   version='1.0',
   description='ML utilities for all',
   author='Not Chat GPT',
   author_email='',
   packages=['AutoML4All'],  #same as name
   install_requires=['numpy==1.23.5', 'pandas==1.5.3', 'scikit-learn==1.2.2','matplotlib==3.7.2','matplotlib-inline==0.1.6','plotly==5.16.1','pycaret==3.0.4','pickleshare==0.7.5','seaborn==0.12.2','streamlit==1.25.0','pygwalker','pandasai'], #external packages as dependencies
)













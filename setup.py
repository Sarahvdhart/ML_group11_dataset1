from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ktml",
    version="0.3",
    description="""Final Project TM10011ML""",
    license="Apache 2.0 License",
    install_requires=required,
    include_package_data=True,
    package_data={
        # Include any *.csv files found within the package
        "worclipo": ['*.csv']
     
    },
    packages=[
        "worclipo"

    ],
)

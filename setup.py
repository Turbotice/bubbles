from setuptools import setup,find_packages

setup(
    name='bubble_codes',
    version='1.0',
    description='Compute velocity fields near interfaces',
#      url='needs a URL',
    author='Alienor Riviere, Kamel Abahri and Stephane Perrard',
    author_email='stephane.perrard@espci.fr',
    license='GNU',
    packages=find_packages(),
    zip_safe=False,
#      package_data={'tangle': ['cl_src/*.cl']})
)

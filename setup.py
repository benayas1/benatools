import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

print('Test')
# Package meta-data.
NAME = 'benatools'
DESCRIPTION = 'Utilities package for XGBoost, CatBoost, LightGBM, Tensorflow and Pytorch'
URL = 'https://github.com/benayas1/benatools'
EMAIL = 'benayas1@gmail.com'
AUTHOR = 'Alberto Benayas'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.60'

# What packages are required for this module to be executed?
REQUIRED = [
          'pandas',
          'numpy',
          'scipy',
          'tensorflow',
          'scikit-learn',
          'torch',
          'efficientnet',
          'efficientnet-pytorch',
          'geffnet',
          'timm',
          'xgboost',
          'catboost',
          'lightgbm',
          'hyperopt',
          'statsmodels',
          'category_encoders'
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

setup(
    name=NAME,
    version=about['__version__'],
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,

    # $ setup.py publish support.
    #cmdclass={'upload': UploadCommand},
)


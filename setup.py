import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="benatools",
    version="0.0.11",
    author="Alberto Benayas",
    author_email="benayas1@gmail.com",
    description="Utilities package for XGBoost, CatBoost, LightGBM, Tensorflow and Pytorch",
    long_description="Utilities package for XGBoost, CatBoost, LightGBM, Tensorflow and Pytorch",
    long_description_content_type="text/markdown",
    url="https://github.com/benayas1/benatools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'tensorflow',
          'scikit-learn',
          'torch',
          'efficientnet',
          'efficientnet-pytorch',
          'timm',
          'xgboost',
          'catboost',
          'lightgbm',
          'hyperopt',
          'statsmodels',
          'category_encoders'
      ],
    python_requires='>=3.6',
)
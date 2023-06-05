from setuptools import setup

setup(name='CityFlowRL',  # name of the package, used like this: `import CityFlowRL`
      version='0.2',
      install_requires=['stable_baselines3', 'cityflow', 'numpy', 'selenium']  # And any other dependencies required
      )

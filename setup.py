from setuptools import setup, find_packages

setup(
    name="malkyriss",
    version="0.1.0",
    packages=["malkyriss"],
    entry_points={
        'console_scripts': [
            'malkyriss=malkyriss.main:main',
            'malkyriss-helper=malkyriss.main:helper',
        ],
    },
    install_requires=[
        # your dependencies here
    ],
)
from setuptools import setup, find_packages

setup(
    name="malkyriss",
    version="0.1.0",
    packages=["malykriss"],
    entry_points={
        'console_scripts': [
            'malykriss=malkyriss.main:main',
            'malykriss-helper=malykriss.main:helper',
        ],
    },
    install_requires=[
        # your dependencies here
    ],
)
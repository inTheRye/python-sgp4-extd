from setuptools import setup


requires = [
    "sgp4>=1.4",
    "tensorflow>=1.4.0",
    "numpy>=1.13.3",
    "pandas>=0.19.2",
]


setup(
    name='sgp4_extd',
    version='0.1',
    description='The pararell sgp4 for analytical propagation of space objects.',
    url='https://github.com/inTheRye/python-sgp4-extd',
    author='Takaaki Tanaka',
    author_email='selene.pace.ima@gmail.com',
    license='MIT',
    keywords=['sgp4', 'concurrent', 'satellite', 'propagation', 'tensorflow'],
    packages=[
        "sgp4_extd",
    ],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)

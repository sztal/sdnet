#!/usr/bin/env python
import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://sdnet.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')


setup(
    name='sdnet',
    version='0.0.0',
    description='Social network generation model combining stochastic block model, Schelling segregation process and preferential attachment.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Szymon Talaga',
    author_email='stalaga@protonmail.com',
    url='https://github.com/sztal/sdnet',
    packages=[
        *find_packages()
        #'sdnet',
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pylint',
        'pytest-pylint',
        'pytest-profiling',
        'pytest-benchmark',
        'pytest-doctestplus',
        'coverage',
        'networkx'
    ],
    test_suite='tests',
    package_dir={'sdnet': 'sdnet'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.15.4'
    ],
    license='MIT',
    zip_safe=False,
    keywords='sdnet',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)

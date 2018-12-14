from setuptools import setup, find_packages

setup(
    name="safe_exploration",
    version="0.0.1",
    author="Torsten Koller, Felix Berkenkamp",
    author_email="fberkenkamp@gmail.com",
    license="MIT",
    packages=find_packages(exclude=['docs']),
    install_requires=['numpy>=1.0,<2',
                      'gpytorch>=0.1.0rc5',
                      'casadi',
                      'GPy',
                      'scikit-learn',
                      'deepdish',
                      'scipy',
                      'matplotlib',
                      'pygame'],
    extras_require={'test': ['pytest>=4,<5',
                             'flake8==3.6.0',
                             'pydocstyle==3.0.0',
                             'pytest_cov>=2.0']},
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
)

from setuptools import setup, find_packages

setup(
    name="safe_exploration",
    version="0.0.1",
    author="Torsten Koller, Felix Berkenkamp",
    author_email="fberkenkamp@gmail.com",
    license="MIT",
    packages=find_packages(exclude=['docs', 'mps']),
    install_requires=['numpy>=1.0,<2',
                      'casadi',
                      'scikit-learn',
                      'scipy'],
    extras_require={'test': ['pytest>=4,<5',
                             'flake8==3.6.0',
                             'pydocstyle==3.0.0',
                             'pytest_cov>=2.0'],
                    'visualization': ['matplotlib',
                             'pygame'],
                    'ssm_gpy':  ['GPy'],
                    'ssm_pytorch': ['gpytorch==0.2.1',
                                    'torch<1.1',
                                    'hessian @ https://github.com/mariogeiger/hessian/tarball/master#egg=hessian']},

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
    package_data={'safe_exploration': ['test/*.npz']}



)

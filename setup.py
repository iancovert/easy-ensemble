import setuptools

setuptools.setup(
    name='easy-ensemble',
    version='0.0.1',
    author='Ian Covert',
    author_email='icovert@cs.washington.edu',
    description='For learning optimal model ensembles.',
    long_description='''
        For learning optimal model ensembles.
    ''',
    long_description_content_type='text/markdown',
    url='https://github.com/iancovert/easy-ensemble/',
    packages=['ensemble'],
    install_requires=[
        'numpy',
        'cvxpy',
        'osqp',
        'sklearn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.6',
)
#!/usr/bin/env python

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()



setup(name='Storms',
      version='0.1',
      description='Tropical Storm analysis tool',
      long_description=readme(),
      url='https://github.com/brey/Storms',
      author='George Breyiannis',
      author_email='gbreyiannis@gmail.com',
      license='EUPL',
      packages=['Storms'],
      classifiers=[
          'Programming Language :: Python',
          'License :: OSI Approved :: EUPL',
          'Operating System :: OS Independent',
          'Development Status :: 4 - Beta',
          'Environment :: Other Environment',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
      ],
      
      install_requires=[
                'numpy',
                'datetime',
                'pandas',
                'glob',
                'netCDF4',
                're',
                'xml',
                'feedparser',
                'urllib',
                'urllib2',
                'bs4',
                'math'
            ],
      include_package_data=True,
      zip_safe=False)
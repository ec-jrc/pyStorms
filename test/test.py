#!/usr/bin/env python

from storm import *

s = Storm()

s.parse_atcf(filename='../test/bal212010.dat')

print s.name
print s.basin
print s.data




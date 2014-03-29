#!/usr/bin/env python
try:
	import numpy as np
except ImportError:
	print "We need numpy to run!"

try:
	import scipy as sp
except ImportError:
	print "We need scipy to run!"

try:
	import matplotlib.pyplot as plt
except ImportError:
	print "We probably need matplotlib!"

try:
	from eoldas_ng import *
except ImportError:
	"eoldas_ng did not install properly!!!"

try:
	from gp_emulator import *
except ImportError:
	print "gp_emulator did not install properly!!!"

try:
	from prosail import *
except ImportError:
	print "PROSAIL did not install properly!!!"

print "Everything installed and ready to go!!!"

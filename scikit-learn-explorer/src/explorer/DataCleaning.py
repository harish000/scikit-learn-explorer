'''
Created on Sep 18, 2016

@author: Sriharish
'''
import re
s = 'IBM is cognitive|IBM is "Cognitive"|ibm is enormous  company with employees with cognitive ability'
barSplit = s.split("|")
print(type(barSplit))

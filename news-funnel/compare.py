import sys

for line1, line2 in zip(open(sys.argv[1]).readlines(), open(sys.argv[2]).readlines()): 
	print line1
	print
	print line2
	print
	print
	raw_input()
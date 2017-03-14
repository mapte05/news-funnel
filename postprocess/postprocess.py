import sys

terminals_file = "non_terminals_list"
min_subsequence_len_allowed = 0  # if 1, output "I like like dogs"; if 0, output "I like dogs"

def main(infile, outfile, remove_bad_terminals=True, remove_repeating_subsequences=True):
    bad_terminals = []
    with open(terminals_file, 'r+') as termf:
    	bad_terminals = [x.strip() for x in termf.readlines()]
    with open(infile, 'r+') as inf:
    	with open(outfile, 'w+') as outf:
    		for summary in [x.strip() for x in inf.readlines()]:
	    		split_summary = summary.split()
	    		if remove_bad_terminals:
	    			if split_summary[-1] in bad_terminals:
	    				summary = split_summary[:-1]
	    		if remove_repeating_subsequences:
	    			for sublen in range(min_subsequence_len_allowed+1, len(split_summary)/2):
	    				for i in range(len(split_summary)-2*sublen+1):
	    					j = 1
	    					while i+sublen*(j+1) < len(split_summary)+1 and split_summary[i:i+sublen] == split_summary[i+sublen*j:i+sublen*(j+1)]:
	    						del split_summary[i+sublen:i+2*sublen]
		    						
	    		outf.write(' '.join(word for word in split_summary)+'\n')


if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    infile = sys.argv[1]
    main(infile, infile+'_postprocessed')
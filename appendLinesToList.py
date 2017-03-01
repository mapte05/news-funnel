'''
Takes file 1 with a JSON list and file 2 with one element per line,
and appends the elements from file 2 to the list in file 1

'''

import json

jsonlist = []

outputfilename = "final_article_list"
inputJSONfilename = 'article_titles'
inputtextfilename = 'article_list'

with open(inputtextfilename, 'r') as f:
    content = f.readlines()
newlist = [x.strip() for x in content if len(x)>0]

with open(inputJSONfilename, 'r') as infile:
    oldlist = json.loads(infile.read())

with open(outputfilename, 'wb') as outfile:
    json.dump(oldlist + newlist, outfile)

print "original list is", len(oldlist), "elements"
print "new list is", len(newlist), "elements"
print "total of", len(oldlist + newlist), 'articles'
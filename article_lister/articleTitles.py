'''
Uses NewsAPI (https://newsapi.org/) to scrape a bunch of news headlines,
put them in a JSON list, and write them to a file.

author: Liam Kinney

'''
import json
import urllib2

outputfilename = 'article_titles'
newsAPIKey = '1b0ffb6751a748f290453746a305240a'
result = []
counter = 0

sourcesJSON = json.loads(urllib2.urlopen("https://newsapi.org/v1/sources").read())
for sourceJSON in sourcesJSON['sources']:
	sourceID = sourceJSON['id']
	url = "https://newsapi.org/v1/articles?source=" + sourceID + '&apiKey=' + newsAPIKey
	articlesListJSON = json.loads(urllib2.urlopen(url).read())
	for articleJSON in articlesListJSON['articles']:
		result.append(articleJSON['title'])
		counter += 1

with open(outputfilename, 'wb') as outfile:
    json.dump(result, outfile)

print counter, "article titles produced"

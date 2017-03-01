from newspaper import Article

for url in open('articles.txt'):
    a = Article(url)
    a.download()
    a.parse()
    print a.text
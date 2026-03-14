# Sentiment Analysis from Webscraping

In this project, we will use the Python libraries Selenium and Beautifulsoup to scrape quotes from famous people from [http://quotes.toscrape.com/](http://quotes.toscrape.com/)

Once we have scraped the quotes from the website, we will use TextBlob and the VADeR library to perform sentiment analysis on these texts.

The project aims to raise awareness of the possibilities offered by web scraping and text sentiment analysis for any business project, such as analysing customer comments on a product.

Throughout the project, we will look at the following steps to follow in order to obtain the data and apply sentiment analysis to famous quotes.

# Contents

- [Libraries](#libraries)
- [Webscraping](#webscraping)
- [Sentiment analysis VADeR](#sentiment-analysis-vader)
- [Visual analysis](#visual-analysis)
- [References](#references)

# Libraries
## Webscraping
```python
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import pandas as pd
import time
```
## Sentiment Analysis
```python
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
```
## Graphs
```python
import matplotlib.pyplot as plt
import seaborn as sns
```
## Graphs configuration
```python
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

# Webscraping

In this section, we will proceed to scrape the aforementioned website. To do so, we will use the Edge webdriver (in my case, I used this webdriver because my version of Chrome is much higher than the current drivers I have installed in Selenium).

The code follows this logic. First, we access http://quotes.toscrape.com/ using the webdriver. Once inside, we begin scraping the quotes. We start with an empty dataframe where we will accumulate the texts of the different authors. First, we search for the quote and author tags to then obtain the author's text and the quote text. We add these values to the empty df that we created at the beginning. Once we have done this with all the quotes on the page, we move on to the next one (we do this by searching for the Next button link on the page and ‘joining’ it to the page URL) and start the loop again. In total, we do this 10 times on this website, as it has a total of 10 pages. s. Finally, after scraping all the quotes, we obtain the dataset with all the relevant information from the website.

```python
driver = webdriver.Edge()
driver.maximize_window()
driver.get("http://quotes.toscrape.com/")

soup = BeautifulSoup(driver.page_source, 'lxml')

df_data = []  # We use a list to accumulate the data

counter = 0

while counter < 10: 
    quote_elements = soup.find_all('div', class_='quote') 
    
    for q in quote_elements:
        quote_tag = q.find('span', class_='text')  # Search for the quote tag
        author_tag = q.find('small', class_='author')  # Search for the author tag
        
        author = author_tag.text if author_tag else ''  # Obtain the author's text if it exists
        quote_text = quote_tag.text if quote_tag else ''  # Get the text of the quote if it exists
        
        df_data.append({'Name': author, 'Quote': quote_text})  # Add the data to the list
    
    next_page_element = soup.find('li', class_='next')
    if next_page_element:
        next_page_link = next_page_element.find('a')['href']    
        next_page_url = urljoin("http://quotes.toscrape.com/", next_page_link)      
        
        page = requests.get(next_page_url)
        soup = BeautifulSoup(page.text, 'lxml')
        time.sleep(2)
    else:
        break  # If there are no more pages, we exit the loop
    
    counter += 1

df = pd.DataFrame(df_data)
df.set_index('Name', inplace=True)
```

# Sentiment analysis VADeR

VADER (Valance Aware Dictionary and sEntiment Reasoner) is a rule-based, lexicon-based sentiment analysis model, specially optimised for short texts such as those found on social media. It uses a predefined dictionary that assigns each word a valence (polarity) score indicating whether it is positive or negative, as well as its intensity. It also considers the context of the words, such as punctuation marks or capitalisation.

For each text, VADER returns four values:
- pos: Proportion of the text that is positive (0 to 1).
- neg: Proportion of text that is negative (0 to 1).
- neu: Proportion of text that is neutral (0 to 1). The sum of the three is 1.
- compound: Normalised score that summarises the overall sentiment on a range from -1 (very negative) to +1 (very positive).

With regard to scoring, the labels used to determine whether a text is positive, negative or neutral are based on the compound result (for example, compound >0 or >0.5 then “positive”), so the labels in different analyses will depend on the analyst's criteria.

```python
sia = SentimentIntensityAnalyzer()

citas = df['Quote']

respuestas_list = []
compound_scores = []
pos_scores = []
neg_scores = []
neutral_scores = []
sentiment_labels = []
for response in citas:
    sentiment_scores = sia.polarity_scores(response)
    respuestas_list.append(response)
    compound_scores.append(sentiment_scores['compound'])
    pos_scores.append(sentiment_scores['pos'])
    neg_scores.append(sentiment_scores['neg'])
    neutral_scores.append(sentiment_scores['neu'])
    sentiment_labels.append('Positive' if sentiment_scores['compound'] > 0 else ('Neutral' if sentiment_scores['compound'] == 0 else 'Negative'))

sentiment_df = pd.DataFrame({
    'Respuestas': respuestas_list,
    'Sentimiento_Compound': compound_scores,
    'Sentiment_pos': pos_scores,
    'Sentiment_neg': neg_scores,
    'Sentiment_neutral': neutral_scores,
    'Sentiment_VADER_label': sentiment_labels})
```

# Sentiment analysis NaiveByaes

It is based on Bayes' theorem, which relates inverse conditional probability. In the context of sentiment analysis, the conditional probabilities that a document belongs to a given sentiment category are calculated based on its combination of words.

P(sentiment|words): the probability that the sentiment is true given a set of words. This is calculated using the inverse conditional probabilities and the ‘naive’ assumption of conditional independence between words. 

Bayes' theorem is a fundamental tool in sentiment analysis and statistics. It allows us to update beliefs about the sentiment of a text based on the evidence present.

Bayes' theorem for sentiment analysis:
- A: Sentiment we are determining (positive, negative, neutral). (in the case of TextBlob, it only assigns two classes, pos and neg)
- B: Characteristics or words in the text.

General formula of Bayes' theorem:
- P(A|B) = (P(B|A) * P(A)) / P(B)

In sentiment analysis:
- P(A|B): Probability of sentiment A given the set of words B.
- P(B|A): Probability of characteristics B given sentiment A.
- P(A): Initial probability of sentiment A before seeing the text.
- P(B): Probability of observing characteristics B in general.

```python
citas = df['Quote'].tolist()

resultados_sentimietno = []
for respuesta in citas:
    blob = TextBlob(respuesta, analyzer = NaiveBayesAnalyzer())
    sentiment = blob.sentiment
    resultados_sentimietno.append((respuesta, sentiment.classification, sentiment.p_pos, sentiment.p_neg))

sentiment_df_NaiveBayes = pd.DataFrame(
    resultados_sentimietno, columns = ['Answers', 'Clasification_NaiveBayes', 'Prob Positive', 'Prob Negative'])


df_final = pd.concat([sentiment_df, sentiment_df_NaiveBayes], axis = 1)
df_final = df_final.drop(['Answers'], axis = 1)
```

# Visual analysis

## Distribution of VADER compound scores

This graph shows the frequency of “Sentiment_Compound” values, which summarise the overall sentiment of each citation on a scale from -1 to +1. The vertical red line at 0 theoretically separates the negative (left) from the positive (right). The image shows that most quotes have a positive compound score (peaking around 0.0–0.1), with a less pronounced tail towards negative values.

```python
plt.figure(figsize=(10, 6))
sns.histplot(df_final['Sentimiento_Compound'], bins=30, kde=True, color='skyblue')
plt.axvline(0, color='red', linestyle='--', label='Neutral (0)')
plt.title('Distribution of VADER Compound Scores')
plt.xlabel('Compound Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

![Distribution of VADER Compound Scores](https://github.com/PabloCH2410/Sentiment-Analysis_from_Webscraping/blob/6d90edefb55d1e172f53f1ddd99a727689d99fcf/Distribution%20of%20VADER%20Compound%20Scores.png)

## Distribution of positive, negative and neutral scores (VADER)

The columns “Sentiment_pos”, “Sentiment_neg” and “Sentiment_neutral” are reorganised to create a box plot comparing the proportions of each word type in the quotes. The boxplot shows that the neutral score is the highest (median ~0.6–0.7), followed by the positive (median ~0.2) and the negative (median ~0.1). This indicates that most words are neutral, with few negative words.

```python
scores_vader = df_final[['Sentiment_pos', 'Sentiment_neg', 'Sentiment_neutral']].melt(var_name='Type', value_name='Score')

plt.figure(figsize=(10, 6))
sns.boxplot(data=scores_vader, x='Type', y='Score', palette='Set2')
plt.title('Distribution of Positive, Negative, and Neutral Scores (VADER)')
plt.ylabel('Score')
plt.show()
```

![Distribution of VADER Compound Scores](https://github.com/PabloCH2410/Sentiment-Analysis_from_Webscraping/blob/9b956fe31545aec5238f68dfee8ee60c2e75de97/Distribution%20of%20Positive%2C%20Negative%2C%20and%20Neutral%20Scores%20(VADER).png)

## Frequency of sentiment labels according to VADER

Counts how many mentions were classified as Positive, Neutral or Negative by VADER. The image shows 49 positive, 29 negative and 22 neutral mentions, confirming the positive bias of the corpus.

```python
plt.figure(figsize=(8, 5))
sns.countplot(data=df_final, x='Sentiment_VADER_label', order=['Positive', 'Neutral', 'Negative'], palette='viridis')
plt.title('VADER Sentiment Classification')
plt.xlabel('Label')
plt.ylabel('Number of Quotes')
plt.show()
```

![Distribution of VADER Compound Scores](https://github.com/PabloCH2410/Sentiment-Analysis_from_Webscraping/blob/37a802ac4fdf1c25953583755a97c66a691775cc/VADER%20Sentiment%20Classification%20distribution.png)

## Frequency of sentiment labels according to NaiveBayes

NaiveBayes only assigns “pos” or “neg” (no neutral). This graph shows the count (69 positive, 31 negative). It can be seen that there are more positive than negative mentions, in a similar proportion to VADER, but VADER’s neutral mentions have been split between the two.

```python
plt.figure(figsize=(8, 5))
sns.countplot(data=df_final, x='Clasification_NaiveBayes', palette='coolwarm')
plt.title('NaiveBayes Sentiment Classification')
plt.xlabel('Label')
plt.ylabel('Number of Quotes')
plt.show()
```


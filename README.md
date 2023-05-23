# text-summarizer-bbc-dataset
In today's digital age, we are constantly bombarded with an overwhelming amount of information, making it challenging to stay informed and up-to-date with the latest news and developments. Traditional news articles can be lengthy and time-consuming to read, making it difficult for busy individuals to keep abreast of the world's top stories.
The goal of this project is to develop a machine learning model that can generate accurate and concise summaries of news articles from various sources, providing individuals with quick and easy-to-digest information about the day's top stories. By summarizing the key points and themes of news articles, this model will facilitate keeping individuals informed and up-to-date with important news from around the world, enabling them to make better-informed decisions.

## Table of Contents

<!-- ⛔️ MD-MAGIC-EXAMPLE:START (TOC:collapse=true&collapseText=Click to expand) -->
<details>
<summary>(click to expand)</summary>
    
  * [Libraries](#libraries)
  * [Getting to know the data](#getting to know the data)
  * [Demo](#demo)
  * [License](#license)  
  * [Disclaimer](#disclaimer)
  * [Questions](#questions)

</details>
<!-- ⛔️ MD-MAGIC-EXAMPLE:END -->

## Libraries

In this we will be using 2 main libraries: lightning and transformers. You can install them as follows

```
pip install lightning
pip install 'transformers[torch]'
```

## Getting to know the data

We have 2225 data points, which consists of 5 categories in total. The five categories we want to identify are Sports, Business, Politics, Tech, and Entertainment.

## Data Cleaning and Preprocessing

Data cleaning and preprocessing are crucial steps in preparing a dataset for text summarization tasks. Here are the steps I followed to clean and preprocess the BBC News dataset for a text summarizer:

### Convert to Lower Case

Lowercasing all text data is a simple yet highly impactful form of text preprocessing that is often underestimated.
```
df['Lower'] = df['Articles'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df.head(5)
```

### Removing Punctuations
```
df['Punctuations_removal'] = df['Lower'].str.replace('[^\w\s]','', regex = True)
df.head(5)
```
### Removing Special Characters
```
df['Special_Characters_removal'] = df['Punctuations_removal'].apply(lambda x: ''.join(re.sub(r"[^a-zA-Z0-9]+", ' ', charctr) for charctr in x ))
df.head(5)
```
### Tokenization

Tokenization is the process of breaking text into smaller units called tokens, typically words or subwords. It helps to structure the data and enables further analysis. For example, the sentence "I love cats" can be tokenized into individual words: ["I", "love", "cats"].
```
from textblob import TextBlob
df['Tokenization'] = df['Stopwords_removal'].apply(lambda x: TextBlob(x).words)
```

### Stemming

Stemming is the process of reducing words to their base or root form by removing suffixes. It aims to normalize words and group together words with similar meanings. For example, stemming the words "running," "runs," and "ran" would all result in the base form "run."
```
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['Stemming'] = df['Tokenization'].apply(lambda x: " ".join([st.stem(word) for word in x]))
```

### Lemmatization

Lemmatization is similar to stemming but aims to transform words into their base form using linguistic analysis and dictionary lookup. Unlike stemming, lemmatization produces valid words. For example, lemmatizing the words "running," "runs," and "ran" would all result in the base form "run."
```
from textblob import Word
df['Lemmatization'] = df['Stemming'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
```

### Removing Ascii Characters and null values from Articles and Summary
```
df['Articles']=df['Articles'].str.encode('ascii','ignore').str.decode('ascii')
df['Summary']=df['Summary'].str.encode('ascii','ignore').str.decode('ascii')
```

## Exploratory Data Analysis (EDA)

After our data pre-processing, we have a data frame which consists of 5 different categories:
['tech', 'sport', 'politics', 'entertainment', 'business']
The columns consisted of 3 features: Category, Articles and Summaries. We had a total of 2225 rows. Then we tokenized the articles and summaries using T5Tokenizer and found out that most of the tokens in articles counted between 0 to 1000 where max is around 500 and most of the tokens in summaries counted between 0 to 500 where max is around 200.

## Data Model

The init method of the class is used to define the architecture of the model, typically by instantiating the various layers of the model and assigning them as attributes of the class.
The forward method is responsible for performing the forward pass of the data through the model. The forward method takes in the input data and applies the model's layers to the input in order to produce the output. The forward method should implement the logic of the model, such as applying the input through a series of layers and returning the output.
The init method of the class instantiates an embedding layer, a transformer layer, and a fully connected layer and assigns them as attributes of the class.
The forward method takes in the input data x , applies the input through the defined layers and returns the output.
When training a transformer model, the training process typically consists of two main steps: the training step and the validation step.
The training_step method defines the logic for performing a single step of training, which typically includes:
●forward pass through the model
●computing the loss
●computing gradients
●updating the model's parameters
The val_step method is similar to the training_step method, but it is used to evaluate the model on a validation set. It typically includes:
●forward pass through the model
●computing the evaluation metrics

## Testing

We assessed manually and also using the ROGUE metrics, where we used 10 test datasets. The ROUGE metrics looks as follows
{'rouge-1': {'r': 0.6416287939944045,
  'p': 0.9226591740615724,
  'f': 0.7451230703978617},
 'rouge-2': {'r': 0.5553065799201777,
  'p': 0.8811679927270706,
  'f': 0.6674467872579355},
 'rouge-l': {'r': 0.6369896988048506,
  'p': 0.9159574880061223,
  'f': 0.73964502719168}}
  
## Conclusion

In conclusion, through the development of a powerful and effective news article summarizer, we be able to reduce information overload and help individuals and organizations make informed decisions with greater ease.
With the potential impacts of leveraging the power of state-of-the-art natural language processing models like T5 and continually refining the technology, we will continue to create tools that not only saves time but also increases accessibility to information.

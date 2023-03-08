# Group ACE project: Difference between left-wing media and right-wing media in the US

## Description
This project aims to study the difference between left-wing media
and right-wing media in the US. More precisely, we want to know how
left-wing media and right-wing media differ in ideology and contents they
report among topics that involve physical conflicts and topics that are not.

Using the word embedding model, LDA and word associations, we can conclude
several findings that supports our hypotheses. Among all the topics, in physical
violence related topic, left and right wing media have low difference in
content coverage, but high difference in ideological context. Moreover, our
analyses provided further divide within physical conflict topics. We recognized
that, war-related physical conflict topics are surprisingly similar in that they
all have low content difference and medium ideology difference. Other physical
conflicts topics that are non-war related are more scattered, but they all tend to
have low content diversity, even though higher than war topics, and high ideological
differences.

## Data Sources:
CNN: https://www.cnn.com/

Nypost: https://nypost.com/

## Required Library:

beautifulsoup4==4.11.2
Django==4.1.7
gensim==4.3.0
jellyfish==0.9.0
matplotlib==3.6.3
newspaper==0.1.0.7
nltk==3.8.1
numpy==1.23.3
pandas==1.4.4
psutil==5.9.0
pyLDAvis==3.4.0
pytest==7.2.1
requests==2.28.1
scikit_learn==1.2.1
scipy==1.9.1
selenium==4.8.2
tqdm==4.64.1
wordcloud==1.8.2.2
smart_open==6.3.0
wordcloud==1.8.2.2
newspaper3k==0.2.8

## Navigation:
You can use this repo to replicate our research and modify it to similar research.

Our whole analysis pipeline can be finished on our demo_config.ipynb. However, for the consideration of time and computation resource, we strongly recommend you to run it on colab or other service platform.

To clean the news data under certain topic, you can run the cells under "Data Collection & Cleaning".
You can follow the instruction and configure kargs_1 to specify the searching
details.

Then, you can train the models based on previously collected data, using the cells under "Embedding Model Training"

Then, you need to train aligning algorithm to align models together. If you have your own pretrained model, you can also load it, just make sure that it have similar attributs and methods as gensim.model.

For the analysis part, you can run the codes under topic description and Content Coverage & Ideological Context
to get the visualization we present in our slides and documents.

If you want to explore the media, topics, or models applied, make sure to adjust alll the details including file naming formulas and stat variables.

## Tasks:
Qichang Zheng: Crawling and Word Clouding
Yutong Jiang: LDA Analysis
Kekun Han: Content CLustering
Zejian Lyu: W2Vec Model, Aligning, and Final Analysis

#### Links：
Doc Link： https://docs.google.com/document/d/1m3is2uHfyMp0Rep0HLwdw_JuZm7M5Naud1zQ9yV0lyk/edit

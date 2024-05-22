# HLT-Sentiment-Analysis: Master's Degree Exam


<div align="center">
<img hight="250" width="400" alt="GIF" align="center" src="https://github.com/Angelido/HLT-Sentiment-Analysis/blob/main/Figures/simp.gif">
</div>

</br>
</br>

We believe that understanding whether a product review is positive or negative, and why it is positive or negative, is useful for several reasons. Firstly, such a system can improve the user experience by providing insights and recommendations on the quality of the product in question. Furthermore, companies can gain valuable insights from customer feedback, enabling them to make informed decisions to improve their products. Finally, an NLP system for sentiment analysis can help companies manage their online reputation more effectively by addressing negative feedback early on and promoting positive engagement with customers.

The aim of our project is therefore to take product reviews collected by Amazon as input and to carry out a twofold task 
- to be able to distinguish between positive and negative reviews (Sentiment Analysis);
- to understand why a negative review is negative (Topic Modeling).

Our idea is to attempt a binary classification using only the titles of the reviews as input, such as "Great CD" or "Batteries are dead within a year". It is clear that the titles are usually short and impactful sentences. Therefore, we believe that they may be sufficient to make our classifier work effectively. We then use the entire reviews instead of just the headlines, focusing only on the negative ones, to carry out the task of topic modelling. This way we hope to be able to tell when a negative review is related to technical problems with the product, related to content issues, related to delivery delays, or related to other problems.

## Main features:
- Uses machine learning and natural language processing techniques.
- Trains a sentiment analysis model on a dataset of Amazon product reviews.
- Classifies reviews as positive or negative based on their sentiment.
- Performs Topic Modelling to understand the causes of negative reviews.

## Technologies Used:
- Python
- PyTorch
- Transformers library (for BERT model)
- NLTK
- Amazon product review dataset

## Contributors:
- [Angelo Nardone](https://github.com/Angelido)
- [Matteo Ziboli](https://github.com/MatteoZb)
- [Riccardo Marcaccio](https://github.com/Riccardo369)

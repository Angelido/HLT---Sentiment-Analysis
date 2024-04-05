# HLT-Sentiment-Analysis: Master's Degree Exam
## By Angelo Nardone, Matteo Ziboli e Riccardo Marcaccio

<div align="center">
<img hight="250" width="400" alt="GIF" align="center" src="https://github.com/Angelido/HLT-Sentiment-Analysis/blob/main/Figures/simp.gif">
</div>

</br>
</br>

We believe that understanding whether a product review is positive or negative is useful for several reasons. Firstly, such a system can enhance user experience by providing insights and recommendations on the quality of the product in question. Additionally, businesses can also gain valuable insights from customer feedback, enabling them to make informed decisions to improve their products. Lastly, an NLP system for sentiment analysis can help companies manage their online reputation more effectively by promptly addressing negative feedback and fostering positive engagement with customers.

So the purpose of our project is to take as input product reviews collected from Amazon (we will discuss the dataset later) and be able to distinguish between positive and negative reviews.

Our idea is to attempt binary classification using only the review titles as input, such as ”Great CD” or ”Batteries died within a year”. Clearly, titles are usually short and impactful phrases. Therefore, we believe that these may be sufficient to effectively operate our classifier. However, we plan to extend this idea by also attempting classification using the entire reviews as input. Using the entire reviews entails processing longer sentences and greater computational effort, but it also provides more information and potentially higher accuracy. At this point, our plan is to compare the results of these two classifiers using various metrics to determine which approach is more effective.

Update: We are including an additional potential task to implement. We have identified a second dataset that again would allow us to input product reviews from Amazon and classify them as positive or negative reviews. In this case, we could evaluate our algorithms on the two different datasets to assess their effectiveness.
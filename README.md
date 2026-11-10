# 506-applied-midterm
# Kaggle Username == Travis Meyer (traviswmeyer)

The initial build of the model had almost no intuition behind it. I didn't see anything clearly jumping out in the data directly, as most of the attributes are not things that are extremely great to look at. Small things like VotedHelpful didn't seem to help the main issue, which is trying to differentiate moderate ratings. 

I began with TF-IDF on the review text. Overtime this would grow from individual words, to small 3-word phrases, in an attempt to pick up patterns. Later on, I tried to used a recommender system, however the returns were minimal and I thought the risk of overfitting by adding even more to the model was too high. 

While developing the model, I ran down the list of tools I knew; SVM was decent but eventually I was pushed to ordinal classification to take advantage of the fact we are looking at ratings (which have order). I then added Naive Bayes-inspired word weighting to train another ordinal classifier. Combining these models helped improve performance. 

Because there are a lot of small choices within the model I created, the majority of my time was spent tuning these parameters. I added a cost matrix, as I was finding my errors within the moderate ratings (2, 3, 4) to be substantial. This also helped as the metric I was using for accuracy was Macro-F1, matching the Kaggle Competition, and this helped improve that score. My final accuracy boost would come from cross validation with folding the training data. This was done completely on a whim from the small bit that I had heard about, and it ended up being very helpful. 

The Model ended up performing very well. While there were a lot of features in the end, they all worked well together. All of my local testing had decent scores, and they were higher in the actual Kaggle submissions. While my top (private) score was 0.56910, my top selected time was 0.56904, as I believed the more general parameters would serve me well (they did not). 

While my end product was good, it was very hard to push the model in the right direction. Because I was looking at methods that are independent, I had multiple (now deleted) models that incorporated different combinations of these methods. While using all of them at once worked the best, I couldn't exactly rule out overfitting, other than keeping good practice when choosing certain parameters I had access to. I initially wanted to use plots to understand how general my model was, but due to the nature of the data it wasn't very useful, and eventually I scraped that. 

At the same time, I feel like there was something I was missing. This could just be inexperience, but none of the improvements I made to the model were that large in aggregate, and I find it hard to believe my model was perfectly fit for the kind of data I was working with.

## Reproducing Submission

The main file used for my final submissions was music_rich_5fold.py. By placing train.csv and test.csv in the same directory, the basic script can run. There are parameters set in the file (immediately after the imports) that effect the actual model in addition to the output files. My best selected submission had the following settings:
N_FOLDS = 5
ALPHA = 2e-6
NB_BLEND = .4
Running the script with MODE = "submit" will result in a csv file (name of the file is also a choice within settings), while "cv" simply runs everything locally and gives metrics on performance. RUN_NAME is very important as it is used so that the model doesn't have to be retrained, however if any parameters are changed the name of the run must also change.

Disclaimer on AI usage and repo hygiene: I used AI extensively, and I tried to delete as little as possible from the final repo. Things like the plots directory, and some of the python has been abandoned, however I kept it to show the progression of the code overtime. The two files that are required to reproduce my code are music_rich_ordinal.py and music_rich_5fold.py. Note that for the references, I used materials to supplement the work I was doing through AI. The fact that I was brute-forcing the learning process is why the references may seem weird.

References:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html 
  - SGD naturally became an option, and was implemented after fitting the SVM. This page was used to doublecheck if it met the requirements, as well as making sure my code notation was correct.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
  - This was used to optimize the cost matrix. This page was used to check notation within the code.
Wang S, Manning C. (2012) Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. 
  - This paper was found while implementing Naive Bayes. This paper outlines how Naive Bayes and SVM work well together, and was the building block of the NB used in the model.
https://sites.google.com/view/ordinalclassification2020/method
  - Used as an informational supplement.
https://arxiv.org/html/2503.00952v1
  - Used as an informational supplement.

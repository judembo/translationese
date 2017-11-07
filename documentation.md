## The corpus

We have a corpus consisting of two parts: The first part contains book chapters of books that were originally written in Hebrew, and the second part contains book chapters that were translated from other languages into Hebrew. Each chapter is annotated with the gender and age of the author, the publisher, gender of the translator (for translations), and the source language (for translations). 
The sentences in the corpus are annotated with a set of morphological and syntactic features, such as lemma, pos tag or binyan.

## Sanity check gender classifier

In a first attempt we are trying to build a classifier that can distinguish between the genders of authors. For now we only have a binary classifier that was trained only on original texts. In a next step we are planning to train a 4-way classifier on the translated data.

* **File**: genderClass\_sanity.py
* **How to run**: 
````
    python genderClass\_sanity.py <corpus_file> <results_file> 
````
 * results_file will contain the labels assigned to the translation data. 0 means 'female' and 1 means 'male'
 * also creates lemmaCounts.txt (counts for top 1000 lemmas in original data), origInstances.txt and translationInstances.txt (contain the sentences of the artificially created instances)
 * F1-score and accuracy will be printed to the command line
 
* **Features**: Relative frequencies of top 300 lemmas (that appear in the originals)
* **Instances**: We create artificial instances for training and testing as follows:
  * Collect all sentences for each gender group separately. This means there are two groups for originals (Male/Female, the classes we want to learn), and four groups for translations (FF, FM, MM, MF)
  * Shuffle all the sentences within each group. We do this to eliminate all other factors such as genre, publisher, source language etc. that we don't want the classifier to learn. We are only interested in distinguishing between genders.
  * For each group, instances are created by selecting sentences until the instance contains 2000 or more tokens.
  * Because in the original data there is more data from male authors than from female authors, we only create as many 'male' training instances as we have 'female' ones to keep the training data balanced.
  * When creating instances of ~2000 tokens we are able to create 135 training instances for each gender 
* **Training/Testing**: The classifier is trained on the artificial instances created from the original data. It is tested on the same instances using 5-fold cross validation. It is also run on the artificial instances created from the translation data. The assigned labels (0 for 'female', 1 for 'male') are saved to the file specified when running the code.
* **Results**:
  * Using the top 300 lemmas and instance sizes of ~2000 tokens, and average F1-score of 0.764695322 is achieved (5-fold cross-validation).
  * The results (F1-score) can vary greatly for each run, as they seem to highly depend on the artificial instances that are created, and they are randomly created with each call of the program. So the above score was calculated by taking the average of 10 calls of the program. For some calls the average F1-score was as high as 0.835, but the lowest was 0.664.
  * When using smaller instances of ~1000 tokens each, we get a lower average F1-score of 0.69766079564
  * When looking at the assigned labels for the translation data (predictedGenders_sanity.txt) we can see that the model seems to have reasonable predictive power on the translations. Still, this varies with every call of the program due to the random creation of instances.
  * F1-scores for labels on translated data still need to be computed


## Gender classifier using morphological/pos features

* **File**: genderClass.py
* **How to run**: 
````
    python genderClass.py <corpus_file> <results_file> 
````
* Works exactly the same way as the sanity check version (instance creation, structure of code), only the used feature set is different.
* **Features**:
  * relative tag counts
  * relative tense counts (normalized over all tokens that have tense)
  * relative binyan counts (normalizes over all tokens that have binyan)
  * relative suffunction counts (normalized over all tokens)
  * relative pos-trigram counts
  * use of prefconj (relative count)
  * use of relativizer (relative count)

* **Results**:
  * Using artificial instances sizes of ~2000 tokens we get an average F1-score of 0.764387377632
  * However when looking at the assigned labels for the translation data (predictedGenders.txt), we can see that the model has poor predictive power as most instances are labelled as 'male', even when both author and translator were female


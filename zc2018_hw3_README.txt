Running the POS Tagger System
Setup
Ensure you have Python 3.x installed on your system. This POS tagger does not require any external libraries beyond the Python Standard Library.

Files Required
WSJ_02-21.pos: Training data in POS-tagged format.
WSJ_24.pos: Development data in POS-tagged format for evaluation.
WSJ_23.words: Test data containing untagged words.
Steps to Run the System
Load the Data: Use the read_data function to load the training, development, and test data. Ensure the data files are in the same directory as the script or provide the correct path.

python
Copy code
training_data = read_data('WSJ_02-21.pos')
development_data = read_data('WSJ_24.pos')
test_data = read_data('WSJ_23.words')
Calculate Probabilities:

Calculate prior probabilities using calculate_prior_probabilities(training_data).
Prepare for OOV handling and calculate likelihoods using calculate_likelihoods(training_data, training_vocabulary, tag_frequencies, total_tag_counts).
Run the Viterbi Algorithm: Use the viterbi_algorithm function to tag the test data.

python
Copy code
tagged_test_data = viterbi_algorithm(test_data, prior_probabilities, likelihoods, oov_likelihoods, unique_tags_list, training_vocabulary)
Output: The tagged sequences can be formatted using format_output(tagged_test_data) and saved to a file.

Evaluate the Model (Optional): Evaluate the model's performance on the development data using evaluate_model.

How to Handle OOV Items
OOV (Out-Of-Vocabulary) items are handled using a combination of strategies:

Categorization Based on Word Features: Words not found in the training vocabulary are categorized based on their features:

Numeric strings are categorized as <NUM>.
Words starting with a capital letter are categorized as <CAP>.
Words ending with 's' are categorized as <PLURAL>.
Strings with only punctuation are categorized as <PUNCT>.
All other OOV words are categorized as <UNK>.
Assigning Probabilities to OOV Categories: Each OOV category is assigned a probability based on the context and characteristics of the word. For example, <NUM> is assigned a high probability for the CD (cardinal number) tag.

Distribution of Items Occurring Once: The likelihood of a word being tagged with a specific POS tag is also based on the distribution of tags among words that occur only once in the training corpus.

Notes
The accuracy of the POS tagger, especially in handling OOV words, is highly dependent on the training data and the effectiveness of the OOV handling strategy.
Fine-tuning the parameters and probabilities assigned to OOV categories can significantly impact the system's performance.
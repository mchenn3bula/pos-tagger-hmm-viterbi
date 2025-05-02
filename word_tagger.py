from collections import defaultdict, Counter
import string
import math

def read_data(file_path):
    # Initialize a list to hold the sentences
    sentences = []
    # Initialize a list to hold the current sentence
    current_sentence = []

    with open(file_path, 'r') as file:
        # Determine the file type based on the extension
        is_pos_file = file_path.endswith('.pos')

        # Iterate over each line in the file
        for line in file:
            # Strip leading and trailing whitespaces
            line = line.strip()

            # If the line is empty, it means the end of the current sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if is_pos_file:
                # For .pos files, split the line into word and tag
                word, tag = line.split('\t')  # Assuming the word and tag are separated by a tab
                current_sentence.append((word, tag))
            else:
                # For .words files, just append the word to the current sentence
                current_sentence.append(line)

        # Don't forget to append the last sentence if it's not empty
        if current_sentence:
            sentences.append(current_sentence)

    return sentences

# Calculate prior probabilities
def calculate_prior_probabilities(training_data):
    bigram_counts = defaultdict(int)
    tag_counts = Counter()
    unique_tags = set(tag for sentence in training_data for _, tag in sentence)
    smoothing_factor = 0.01  # Example smoothing factor

    for sentence in training_data:
        tags = [tag for word, tag in sentence]
        for i in range(len(tags) - 1):
            bigram_counts[(tags[i], tags[i + 1])] += 1
            tag_counts[tags[i]] += 1
        tag_counts[tags[-1]] += 1

    bigram_probabilities = defaultdict(float)
    for (tag1, tag2), count in bigram_counts.items():
        bigram_probabilities[(tag1, tag2)] = (count + smoothing_factor) / (tag_counts[tag1] + smoothing_factor * len(unique_tags))

    return bigram_probabilities


def calculate_likelihoods(training_data, training_vocabulary, tag_frequencies, total_tag_counts):
    word_counts_per_tag = defaultdict(Counter)
    tag_counts = Counter()

    for sentence in training_data:
        for word, tag in sentence:
            word_counts_per_tag[tag][word] += 1
            tag_counts[tag] += 1

    likelihoods = defaultdict(dict)
    vocab_size = len(training_vocabulary)

    # Calculate likelihoods for words in the training vocabulary
    for tag, word_counts in word_counts_per_tag.items():
        for word in word_counts:
            if word in training_vocabulary:
                # Add-one smoothing
                likelihoods[tag][word] = (word_counts[word] + 1) / (tag_counts[tag] + vocab_size)

    # Calculate and store OOV likelihoods separately
    oov_likelihoods = defaultdict(dict)
    for tag in tag_counts:
        # Calculate the likelihood for OOV words based on tag frequencies
        oov_likelihood = 1 / (tag_counts[tag] + vocab_size)
        for oov_word in {'<NUM>', '<CAP>', '<PLURAL>', '<PUNCT>', '<UNK>'}:
            oov_likelihoods[tag][oov_word] = oov_likelihood

        # Use distribution of items occurring once for general OOV words
        oov_likelihoods[tag]['<OOV>'] = tag_frequencies.get(tag, 0) / total_tag_counts.get(tag, 1)

    return likelihoods, oov_likelihoods


def handle_oov_words(word, training_vocabulary, tag_frequencies, total_tag_counts):
    if word in training_vocabulary:
        return {tag: 1/len(training_vocabulary) for tag in total_tag_counts}

    likelihoods = {tag: 1/1000 for tag in total_tag_counts}  # Default small likelihood

    if word.isdigit():
        likelihoods['CD'] = 1.0
    elif word[0].isupper():
        likelihoods['NNP'] = 0.5
    elif word.endswith('s'):
        likelihoods['NNS'] = 0.5
    elif all(c in string.punctuation for c in word):
        likelihoods['PUNCT'] = 0.5
    else:
        # Use distribution of items occurring once
        for tag, count in tag_frequencies.items():
            likelihoods[tag] = count / total_tag_counts[tag]

    return likelihoods

def viterbi_algorithm(test_data, prior_probabilities, likelihoods, oov_likelihoods, unique_tags, training_vocabulary):
    tagged_sequences = []
    log_zero = -float('inf')
    smoothing_value = 1e-10  # A small value to avoid log(0)

    for sentence in test_data:
        n = len(sentence)
        m = len(unique_tags)

        viterbi = [[log_zero] * m for _ in range(n)]
        backpointer = [[0] * m for _ in range(n)]

        # Initialization
        for i, tag in enumerate(unique_tags):
            word = sentence[0]
            emission_prob = likelihoods[tag].get(word, oov_likelihoods[tag].get(word, smoothing_value))
            transition_prob = prior_probabilities.get(('<s>', tag), smoothing_value)

            # Check for zero probabilities and adjust
            transition_prob = max(transition_prob, smoothing_value)
            emission_prob = max(emission_prob, smoothing_value)

            viterbi[0][i] = math.log(transition_prob) + math.log(emission_prob)
            backpointer[0][i] = 0

        # Recursion
        for t in range(1, n):
            for i, tag in enumerate(unique_tags):
                max_prob, max_idx = log_zero, 0
                word = sentence[t]
                emission_prob = likelihoods[tag].get(word, oov_likelihoods[tag].get(word, smoothing_value))
                for j, prev_tag in enumerate(unique_tags):
                    transition_prob = prior_probabilities.get((prev_tag, tag), smoothing_value)

                    # Check for zero probabilities and adjust
                    transition_prob = max(transition_prob, smoothing_value)
                    emission_prob = max(emission_prob, smoothing_value)

                    prob = viterbi[t - 1][j] + math.log(transition_prob) + math.log(emission_prob)
                    if prob > max_prob:
                        max_prob, max_idx = prob, j
                viterbi[t][i] = max_prob
                backpointer[t][i] = max_idx

        # Backtrack
        path = [0] * n
        max_prob, max_idx = max((viterbi[n - 1][i], i) for i in range(m))
        path[n - 1] = unique_tags[max_idx]

        for t in range(n - 2, -1, -1):
            max_idx = backpointer[t + 1][max_idx]
            path[t] = unique_tags[max_idx]

        tagged_sequence = list(zip(sentence, path))
        tagged_sequences.append(tagged_sequence)

    return tagged_sequences


# Evaluate the model
def evaluate_model(development_data, prior_probabilities, likelihoods, unique_tags):
    correct, total = 0, 0

    # Get the predicted sequences
    predicted_sequences = viterbi_algorithm(development_data, prior_probabilities, likelihoods, unique_tags)

    # Compare the predicted sequences with the true sequences
    for predicted, true in zip(predicted_sequences, development_data):
        for (pred_word, pred_tag), (true_word, true_tag) in zip(predicted, true):
            if pred_tag == true_tag:
                correct += 1
            total += 1

    # Calculate and return the accuracy
    accuracy = correct / total if total > 0 else 0
    return accuracy

def format_output(tagged_sequences):
    """
    Format the tagged sequences into a string suitable for saving to a file.
    Each word and its tag are separated by a tab. Each word is on a new line, 
    and sentences are separated by empty newlines.
    """
    output_lines = []
    for sentence in tagged_sequences:
        # Format each word and tag pair, separated by a tab
        for word, tag in sentence:
            output_lines.append(f"{word}\t{tag}")
        # Add an empty line to separate sentences
        output_lines.append("")
    return "\n".join(output_lines)

# input files
training_data = read_data('WSJ_02-21.pos')
development_data = read_data('WSJ_24.pos')
test_data = read_data('WSJ_23.words')

prior_probabilities = calculate_prior_probabilities(training_data)
training_vocabulary = set(word for sentence in training_data for word, _ in sentence)
word_counts = Counter(word for sentence in training_data for word, _ in sentence)
single_occurrence_words = {word for word, count in word_counts.items() if count == 1}
tag_frequencies = Counter(tag for sentence in training_data for word, tag in sentence if word in single_occurrence_words)
total_tag_counts = Counter(tag for sentence in training_data for _, tag in sentence)
likelihoods, oov_likelihoods = calculate_likelihoods(training_data, training_vocabulary, tag_frequencies, total_tag_counts)
unique_tags_list = list(set(tag for sentence in training_data for _, tag in sentence))
tagged_test_data = viterbi_algorithm(test_data, prior_probabilities, likelihoods, oov_likelihoods, unique_tags_list, training_vocabulary)
formatted_output = format_output(tagged_test_data)

# Save the output to a file
with open('submission.pos', 'w') as file:
    file.write(formatted_output)

print("Tagging complete. Output saved to submission.pos")


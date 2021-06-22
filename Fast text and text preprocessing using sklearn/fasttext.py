import torch
import numpy as np

import collections
import os


import nltk
from nltk import word_tokenize
nltk.download('punkt')

pad_word_id = 0
unknown_word_id = 1


def create_label_vec(label):
    """Create one hot representation for the given label.

    Args:
        label(str): class name
    """
    label_to_id = {'World': 0, 'Entertainment': 1, 'Sports': 2}
    label_id = label_to_id[label.strip()]
    return label_id


def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    """ Map a word sequence to a word ID sequence.

    Args:
        token_seq (list) : a list of words, each word is a string.
        word_to_id (dictionary) : map word to its id.
    """
    return [map_word_to_id(word_to_id, word) for word in token_seq]


def map_word_to_id(word_to_id, word):
    """ Map a word to its id.

        Args:
            word_to_id (dictionary) : a dictionary mapping words to their ids.
            word (string) : a word.
    """
    if word in word_to_id:
        return word_to_id[word]
    else:
        return unknown_word_id


def build_vocab(sens_file_name):
    """ Build a vocabulary from a train set.

        Args:
            sens_file_name (string) : the file path of the training sentences.
    """
    data = []
    with open(sens_file_name) as f:
        for line in f.readlines():
            tokens = word_tokenize(line)
            data.extend(tokens)
    print('number of sequences is %s. ' % len(data))
    count = [['$PAD$', pad_word_id], ['$UNK$', unknown_word_id]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)

    print("PAD word id is %s ." % word_to_id['$PAD$'])
    print("Unknown word id is %s ." % word_to_id['$UNK$'])
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id


def read_labeled_dataset(Dataset, sens_file_name, label_file_name, word_to_id):
    """ Read labeled dataset.

        Args:
            sens_file_name (string) : the file path of sentences.
            label_file_name (string) : the file path of sentence labels.
            word_to_id (dictionary) : a dictionary mapping words to their ids.
    """
    with open(sens_file_name) as sens_file, open(label_file_name) as label_file:
        data = []
        data_labels = []
        for label in label_file:
            sens = sens_file.readline()
            word_id_seq = map_token_seq_to_word_id_seq(word_tokenize(sens), word_to_id)
            if len(word_id_seq) > 0:
                data.append(word_id_seq)
                data_labels.append(create_label_vec(label))
        print("read %d sentences from %s ." % (len(data), sens_file_name))
        labeled_set = Dataset(sentences=data, labels=data_labels)

        return labeled_set


def write_results(test_results, result_file):
    """ Write predicted labels into file.

        Args:
            test_results (list) : a list of predictions.
            result_file (string) : the file path of the prediction result.
    """
    with open(result_file, mode='w') as f:
        for r in test_results:
            f.write("%d\n" % r)


class Dataset:
    """
        The class for representing a dataset.

    """

    def __init__(self, sentences, labels=None, max_length=-1):
        """
            Args:
                sentences (list) : a list of sentences. Each sentence is a list of word ids.
                labels (list) : a list of label representations. Each label is represented by one-hot vector.
        """
        if labels is None:
            labels = [0] * len(sentences)
            self.has_labels = False
        else:
            self.has_labels = True
        if len(sentences) != len(labels):
            raise ValueError(
                "The size of sentences {} does not match the size of labels {}. ".format(len(sentences), len(labels)))
        if len(labels) == 0:
            raise ValueError("The input is empty.")
        self.sentences = sentences
        self.labels = labels
        self.sens_lengths = [len(sens) for sens in sentences]
        if max_length == -1:
            self.max_sens_length = max(self.sens_lengths)
        else:
            self.max_sens_length = max_length
        self.masks = np.stack(
            [np.concatenate((np.ones(l), np.zeros(self.max_sens_length - l))) for l in self.sens_lengths], 0)
        self.vocab_size = max([max(sent) for sent in self.sentences]) + 1

    def label_tensor(self):
        """
            Return the label matrix of a batch.
        """
        return torch.LongTensor(np.array(self.labels))

    def sent_tensor(self):
        """
            Return the sentence matrix of a batch.
        """
        return torch.LongTensor(np.array(self.sentences))

    def sens_length(self):
        """
            Return a vector of sentence length for a batch.
        """
        length_array = np.array(self.sens_lengths, dtype=np.float32)
        return torch.FloatTensor(np.reshape(length_array, [len(self.sens_lengths), 1]))

    def subset(self, index_list):
        """ Return a subset of the dataset.

            Args:
                index_list (list) : a list of sentence index.
        """
        sens_subset = []
        labels_subset = []
        for index in index_list:
            if index >= len(self.sentences):
                raise IndexError("index {} is larger than or equal to the size of the dataset {}.".format(index, len(
                    self.sentences)))
            sens_subset.append(self.sentences[index])
            labels_subset.append(self.labels[index])

        dataset = Dataset(sentences=sens_subset, labels=labels_subset, max_length=self.max_sens_length)

        return dataset

    def get_batch(self, index_list):
        """ Return a batch.

            Args:
                index_list (list) : a list of sentence index.
        """
        data_subset = self.subset(index_list)
        data_subset.sentences = [self.pad_sentence(s) for s in data_subset.sentences]

        return data_subset

    def pad_sentence(self, sens):
        """ Implement padding here.

            Args:
                sens (list) : a list of word ids.
        """
        return sens + ([0] * max(self.max_sens_length - len(sens), 0))

    def size(self):
        return len(self.sentences)


class DataIter:
    """ An iterator of an dataset instance.

    """

    def __init__(self, dataset, batch_size=1):
        """
            Args:
                dataset (Dataset) : an instance of Dataset.
                batch_size (int) : batch size.
        """
        self.dataset = dataset
        self.dataset_size = len(dataset.sentences)
        self.shuffle_indices = np.arange(self.dataset_size)
        self.batch_index = 0
        self.batch_size = batch_size
        self.num_batches_per_epoch = int(self.dataset_size / float(self.batch_size))

    def __iter__(self):
        return self

    def next(self):
        """ return next instance. """

        if self.batch_index < self.dataset_size:
            i = self.shuffle_indices[self.batch_index]
            self.batch_index += 1
            return self.dataset.get_batch([i])
        else:
            raise StopIteration

    def next_batch(self):
        """ return indices for the next batch. Useful for minibatch learning."""

        if self.batch_index < self.num_batches_per_epoch:
            start_index = self.batch_index * self.batch_size
            end_index = (self.batch_index + 1) * self.batch_size
            self.batch_index += 1
            return self.dataset.get_batch(self.shuffle_indices[start_index: end_index])
        else:
            raise StopIteration

    def has_next(self):

        return self.batch_index < self.num_batches_per_epoch

    def shuffle(self):
        """ Shuffle the data indices for training"""

        self.shuffle_indices = np.random.permutation(self.shuffle_indices)
        self.batch_index = 0


def train_fast_text(model, train_dataset, dev_dataset, test_dataset, model_file_path, batch_size=10, num_epochs=1):
    """ Train a fasttext model, evaluate it on the validation set after each epoch,
    and choose the best one model to evaluate it on the test set.

    Args:
        word_to_id (dictionary) : word to id mapping.
        train_dataset (Dataset) : labeled dataset for training.
        dev_dataset (Dataset) : labeled dataset for validation.
        test_dataset (Dataset) : labeled dataset for test.
        model_file_path (string) : model file path.
        batch_size (int) : the number of instances in a batch.

    """
    max_accu = 0
    max_accu_epoch = 0
    dataIterator = DataIter(train_dataset, batch_size)
    optimizer = model.optimizer
    for epoch in range(num_epochs):
        dataIterator.shuffle()
        total_loss = 0
        n_batches = 0
        # modify here to use batch training
        while dataIterator.has_next():
            batch_data = dataIterator.next_batch()
            sens = batch_data.sent_tensor()
            labels = batch_data.label_tensor()
            sens_lengths = batch_data.sens_length()

            predictions = model(sens, sens_lengths)
            loss = model.loss_func(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        model_file = '{}_{}'.format(model_file_path, epoch)
        if not os.path.exists(model_file_path):
            os.mkdir(model_file_path)
        torch.save(model, model_file)
        validation_accuracy = compute_accuracy(model_file, dev_dataset)
        print('Epoch %d : train loss = %s , validation accuracy = %s .' % (epoch, total_loss / n_batches, validation_accuracy))
        if validation_accuracy > max_accu:
            max_accu = validation_accuracy
            max_accu_epoch = epoch

        # modify here to use batch evaluation
    final_model_file = '{}_{}'.format(model_file_path, max_accu_epoch)
    if test_dataset.has_labels:
        print('Accuracy on the test set : %s.' % compute_accuracy(final_model_file, test_dataset))
    _, predictions = predict(final_model_file, test_dataset)
    write_results(torch.argmax(predictions, 1), model_file_path + 'predictions.csv')


def predict(model_file, dataset):
    """
    Predict labels for each sentence in the test_dataset.

    Args:
        fast_text (FastText) : an instance of fasttext model.
        fasttext_model_file (string) : file path to the fasttext model.
        test_dataset (Dataset) : labeled dataset to generate predictions.
    """

    predictions = []
    labels = []
    model = torch.load(model_file)
    dataIterator = DataIter(dataset)
    while dataIterator.has_next():
        data_record = dataIterator.next()
        sens = data_record.sent_tensor()
        ls = data_record.label_tensor()
        if ls is None:
            ls = []
        sens_length = data_record.sens_length()
        p = model(sens, sens_length)
        predictions.append(p)
        labels.append(ls)
    return torch.cat(labels), torch.cat(predictions)


def compute_accuracy(fasttext_model_file, eval_dataset):
    """
    Compuate accuracy on the eval_dataset in the batch mode. It is useful only for the bonus assignment.

    Args:
        fast_text (FastText) : an instance of fasttext model.
        fasttext_model_file (string) : file path to the fasttext model.
        eval_dataset (Dataset) : labeled dataset for evaluation.
    """
    labels, predictions = predict(fasttext_model_file, eval_dataset)
    acc = (labels == torch.argmax(predictions, 1)).float().mean().item()

    return acc


def load_question_2_1(data_folder):
    """
    Train and evaluate the fasttext model.

    Args:
        data_folder (string) : the path to the data folder.

    """
    trainSensFile = os.path.join(data_folder, 'sentences_train.txt')
    devSensFile = os.path.join(data_folder, 'sentences_dev.txt')
    testSensFile = os.path.join(data_folder, 'sentences_test.txt')
    trainLabelFile = os.path.join(data_folder, 'labels_train.txt')
    devLabelFile = os.path.join(data_folder, 'labels_dev.txt')
    testLabelFile = os.path.join(data_folder, 'labels_test.txt')

    word_to_id = build_vocab(trainSensFile)
    train_dataset = read_labeled_dataset(Dataset, trainSensFile, trainLabelFile, word_to_id)
    dev_dataset = read_labeled_dataset(Dataset, devSensFile, devLabelFile, word_to_id)
    test_dataset = read_labeled_dataset(Dataset, testSensFile, testLabelFile, word_to_id)
    return word_to_id, train_dataset, dev_dataset, test_dataset

from gtd.utils import bleu
import numpy as np

class GrllMetrics:

    @classmethod
    def compute_bleu_score(cls, generated_sentences, original_sentences):
        """ Compute the bleu score between two list of sentences

        Args
            generated_sentences: A list of str representing a generated sentence
            original_sentences: A list of str representing the original sentence

        Returns
            avg_bleu: the average bleu score for the given dataset.
        """
        bleus = []
        for original_sentence, generated_sentence in zip(original_sentences, generated_sentences):
            bleus.append(bleu(original_sentence.split(" "), generated_sentence.split(" ")))
        avg_bleu = np.mean(bleus)
        return avg_bleu

    @classmethod
    def compute_perplexity(cls, sentence_probs, n_list):
        """Compute the perplexity of a generated dataset.

        Args
            sentence_probs: a list of probs for each sentences in a dataset
            n_list: a list of number of token for each sentences in a dataset

        Returns
            pp: the perplexity value of the generated dataset
        """
        logq_list = []
        for sentence_prob in sentence_probs:
            logq_list.append(np.log2(sentence_prob) + 1e-45)

        entropy = - np.sum(logq_list) / np.sum(n_list)
        perplexity = 2.0 ** entropy
        return perplexity




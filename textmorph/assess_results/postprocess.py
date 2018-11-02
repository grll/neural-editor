import logging


class GrllPostprocessor:
    """ Postprocess the data generated.

    Attributes
        forbidden_tokens: list of tokens to remove from generation
        possible_entities: a list of token considered as named entity
    """

    def __init__(self):
        """ Initialize attributes. """
        self.forbidden_tokens = [u"\n", u"\t", u"''", u"(", u")"]
        self.possible_entities = [u"<loc>", u"<org>", u"<ordinal>", u"<per>", u"<unk>"]

    def postprocess_filter(self, edit_traces, min_number_of_token):
        """ Filter the edit_traces coming out of the editions:
        - Remove forbidden tokens
        - Filter Candidates Sequence with less than min_number_of_token
        - Filter Candidates that already exist
        - Filter Candidates that is same as the source sentence

        Args:
            edit_traces: the edit_traces coming from an edition
            min_number_of_token: the minimum number of token to keep a candidate

        Returns:
            list of filtered candidate (dict {sentence, sequence, prob})
        """
        logging.debug("Starting the postprocess filtering on {} edit_traces.".format(len(edit_traces)))
        new_candidates = []
        sequences = []
        for edit_trace in edit_traces:
            for candidate in edit_trace.decoder_trace.candidates:
                logging.debug("Currently filtering: {}.".format(" ".join(candidate.sequence).encode("utf8")))
                new_sequence = []
                for token in candidate.sequence:
                    if token not in self.forbidden_tokens:
                        new_sequence.append(token)
                logging.debug("With forbidden tokens out: {}.".format(" ".join(new_sequence).encode("utf8")))
                if len(new_sequence) < min_number_of_token:
                    logging.debug("The candidate was discarded because it's len was < {}".format(min_number_of_token))
                    break  # forget about this sequence

                if new_sequence not in sequences and new_sequence != edit_trace.example.source_words:
                    if self.entities_check(new_sequence, edit_trace.example.entities):
                        new_candidate = {"sentence": edit_trace.example, "sequence": new_sequence, "prob": candidate.prob}
                        new_candidates.append(new_candidate)
                        sequences.append(new_sequence)
                        logging.debug("The candidate sequence was successfully added to the new_candidates.")
                    else:
                        logging.debug("The candidate sequence didn't pass entities_check.")
                else:
                    logging.debug("The candidate was either similar to the source or already in the saved sequences.")

        return new_candidates

    def entities_check(self, token_sequence, entities):
        """
        Check that the candidate sequence dont have more named entity than the preprocessed sequence (for replacement)

        Args
            token_sequence: a list of tokens in text format (unicode) to check.
            entities: a dict of entities to check accordingly.
        """
        entities_count = {k: len(v) for k, v in entities.items()}
        for token in token_sequence:
            for special_token in self.possible_entities:
                if token == special_token:
                    if special_token not in entities_count:
                        return False
                    else:
                        entities_count[special_token] -= 1

        for _, v in entities_count.items():
            if v < 0:
                return False

        return True

    def generate_n_sentences(self, results, n=100):
        """ Generate n sentences from results.

        Args:
            results: a dictionary of candidate results with keys (generated, prob, sentence)
            n: an int representing the number of sentences to generate

        Returns:
            generated_sentences, original_sentences: a list of generated sentences and its original sentences
        """
        generated_sentences = []
        original_sentences = []
        for result in results[:n]:
            logging.debug("Original Generated sentence: {}".format(" ".join(result["sequence"]).encode("utf8")))
            token_list = self.replace_entities(result["sequence"], result["sentence"].entities)
            logging.debug("Generated sentence after entities replacement: {}".format(" ".join(token_list).encode("utf8")))
            token_list = self.applies_capitalization(token_list, result["sentence"].original_sentence)
            logging.debug("Generated sentence after capitalization: {}".format(" ".join(token_list).encode("utf8")))
            generated_sentences.append(" ".join(token_list))
            original_sentences.append(result["sentence"].original_sentence)
        return generated_sentences, original_sentences

    def replace_entities(self, token_list, entities):
        """ Replace the entities in the token list. A check / filtering postprocess phase should have been performed before.

        Args:
            token_list: the list of token in which to replace entities.
            entities: the listo of entities to replace with.

        Returns:
            new_token_list: a text token list with replaced entities.
        """
        new_token_list = []
        entities_usage = {}
        for k, _ in entities.items():
            entities_usage[k] = 0
        for token in token_list:
            if token in self.possible_entities:
                new_token_list.append(entities[token][entities_usage[token]])
                entities_usage[token] += 1
            else:
                new_token_list.append(token)
        return new_token_list

    def applies_capitalization(self, token_list, original_sentence):
        """ Applies capitalization on the given token_list by trying to find in in the original sentence.

        Args:
            token_list: A list of text token to compare with the original sentence.
            original_sentence: An original sentence to compare with.

        Returns:
            new_token_list: A list of text token with hopefully better capitalization.
        """
        new_token_list = []
        original_token_list = original_sentence.split(" ")
        for idx, token in enumerate(token_list):
            if idx < len(original_token_list):
                if token.lower() == original_token_list[idx].lower():
                    new_token_list.append(original_token_list[idx])
                else:
                    new_token_list.append(token)
            else:
                new_token_list.append(token)

        return new_token_list


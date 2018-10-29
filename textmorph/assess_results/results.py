class Results():

    def __init__(self, edit_traces=[]):
        self.edit_traces = edit_traces
        self.possible_entities = [ u"<loc>", u"<org>", u"<ordinal>", u"<per>", u"<unk>"]

    def add_edit_traces(self, edit_traces):
        self.edit_traces += edit_traces

    def compute_best_n_results(self, n=5):
        """ Compute best n for each sentence using only one generation per edit vector"""
        filtered_results = []
        candidates = []

        for edit_trace in self.edit_traces:
            for candidate in edit_trace.decoder_trace.candidates:
                if (candidate.sequence != edit_trace.example.source_words and
                        candidate.sequence not in candidates and
                        candidate.sequence != edit_trace.example.preprocessed_sentence.split()):

                    if self.entities_check(candidate.sequence, edit_trace.example.entities):
                        filtered_results.append({"example": edit_trace.example,
                                                 "generated": candidate.sequence,
                                                 "prob": candidate.prob})
                        candidates.append(candidate.sequence)
                        break  # once we found the best from these candidates we break the loop to evaluate next random editions

        sorted_filtered_results = sorted(filtered_results, key=lambda x: x["prob"], reverse=True)
        return sorted_filtered_results[:n]

    def compute_overall_best_n_results(self, n=1000):
        """ Compute overall best n for each sentence using multiple generation per edit vector"""
        filtered_results = []
        candidates = []

        for edit_trace in self.edit_traces:
            for candidate in edit_trace.decoder_trace.candidates:
                if (candidate.sequence != edit_trace.example.source_words and
                        candidate.sequence not in candidates and
                        candidate.sequence != edit_trace.example.preprocessed_sentence.split()):

                    if self.entities_check(candidate.sequence, edit_trace.example.entities):
                        clean_candidate_sequence = self.clean_candidate_sequence(candidate.sequence)
                        if len(clean_candidate_sequence) >= 4:
                            filtered_results.append({"example": edit_trace.example,
                                                     "generated": self.clean_candidate_sequence(candidate.sequence),
                                                     "prob": candidate.prob})
                            candidates.append(candidate.sequence)

        sorted_filtered_results = sorted(filtered_results, key=lambda x: x["prob"], reverse=True)
        return sorted_filtered_results[:n]

    def clean_candidate_sequence(self, candidate_sequence):
        sequence = []
        for token in candidate_sequence:
            if token not in [u"\n", u"\t", u"''"]:
                sequence.append(token)
        return sequence

    def entities_check(self, token_sequence, entities):
        """Check that the candidate sequence dont have more named entity than the preprocessed sequence (for replacement)"""
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

    def generate_sentences_without_entities(self, results):
        generated_sentences = []
        for result in results:
            generated_sentence = result['generated']
            entities_usage = {}
            for k, _ in result['example'].entities.items():
                entities_usage[k] = 0

            for idx, token in enumerate(generated_sentence):
                if token in self.possible_entities:
                    generated_sentence[idx] = result['example'].entities[token][entities_usage[token]]
                    entities_usage[token] += 1
            generated_sentences.append(" ".join(generated_sentence))
        return generated_sentences

    def sentences_without_entities_and_back_original_capitalization(self, results):
        generated_sentences = []
        for result in results:
            generated_sentence = result['generated']
            entities_usage = {}
            for k, _ in result['example'].entities.items():
                entities_usage[k] = 0

            for idx, token in enumerate(generated_sentence):
                if token in self.possible_entities:
                    generated_sentence[idx] = result['example'].entities[token][entities_usage[token]]
                    entities_usage[token] += 1
                elif idx < len(result['example'].original_sentence.split(" ")):
                    if token.lower() == result['example'].original_sentence.split(" ")[idx].lower():
                        generated_sentence[idx] = result['example'].original_sentence.split(" ")[idx]
            generated_sentences.append(" ".join(generated_sentence))
        return generated_sentences

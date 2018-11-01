
class GrllPostprocessor:

    def __init__(self):
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
        new_candidates = []
        sequences = []
        for edit_trace in edit_traces:
            for candidate in edit_trace.decoder_trace.candidates:
                new_sequence = []
                for token in candidate.sequence:
                    if token not in self.forbidden_tokens:
                        new_sequence.append(token)
                if len(new_sequence) < min_number_of_token:
                    break  # forget about this sequence

                if new_sequence not in sequences and new_sequence != edit_trace.example.source_words:
                    if self.entities_check(new_sequence, edit_trace.example.entities):
                        new_candidate = {"sentence": edit_trace.example, "sequence": new_sequence, "prob": candidate.prob}
                        new_candidates.append(new_candidate)
                        sequences.append(new_sequence)

        return new_candidates

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

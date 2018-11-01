class GrllResults:

    def __init__(self, candidates=None):
        if candidates is None:
            self.candidates = list()
        else:
            self.candidates = candidates

    def __getitem__(self, item):
        return self.candidates[item]

    def __len__(self):
        return len(self.candidates)

    def add_candidates(self, candidates):
        self.candidates += candidates

    def sort(self,):
        """ Sort the candidates stored in results"""
        self.candidates = sorted(self.candidates, key=lambda x: x["prob"], reverse=True)

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

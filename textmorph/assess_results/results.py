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

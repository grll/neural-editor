from textmorph.edit_model.training_run import EditExample

class Sentence(EditExample):
    """Wrapper arround the EditExample class to add entities, original_sentence and preprocessed_sentences attributes"""

    def __new__(cls, source_words, insert_words=[], insert_exact_words=[], delete_words=[],
                delete_exact_words=[], target_words=[], edit_embed=None, entities={},
                original_data=None, preprocessed_sentence=None):
        if edit_embed is not None:
            assert len(edit_embed.shape) == 1  # must be 1D array
        self = super(EditExample, cls).__new__(cls, source_words, insert_words, insert_exact_words, delete_words,
                                               delete_exact_words, target_words, edit_embed)

        self.entities = entities
        self.original_data = original_data
        self.preprocessed_sentence = preprocessed_sentence
        return self

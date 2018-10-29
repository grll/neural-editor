import spacy
nlp = spacy.load('de_core_news_sm')

class GrllPreprocessor:
    """ Define all the preprocessing on sentence for the neural editor. """

    def __init__(self, word_vocab):
        if word_vocab:
            self.word_vocab = word_vocab

    def preprocess(self, sentence):
        """ Preprocess a given sentence.

        Args:
            sentence: the original sentence to preprocess.

        Returns:
            dict_entities, preprocessed_sentence
        """
        doc = nlp(sentence)
        tokens_with_ents_replaced, entities = self.replace_entities(doc)
        return entities, " ".join(tokens_with_ents_replaced)

    def replace_entities(self, token_list):
        """ Replace the entities in the token list by their named entities and store the entities in a dict.

        Args:
            token_list: A list of spacy token or a doc.

        Returns:
            new_text_token_list, entities: a list of string token coressponding to the preprocessed sentence and a dict
                                           of entities.
        """
        new_text_token_list = []
        entities = {}
        for token in token_list:
            if token.ent_iob_ == "B":
                ent_name = token.ent_type_.lower()

                t = [token.text]
                for token in token.doc[token.idx + 1:]:
                    if token.ent_iob_ == "I":
                        t.append(token.text)
                    else:
                        break
                if ent_name == "misc":
                    for tt in t:
                        if self.word_vocab.word2index(tt) == 0:
                            new_text_token_list.append(u"<unk>")
                            if u"<unk>" in entities:
                                entities[u"<unk>"].append(tt)
                            else:
                                entities[u"<unk>"] = [tt]
                        else:
                            new_text_token_list.append(tt)
                else:
                    new_text_token_list.append("<" + ent_name + ">")

                    if ent_name in entities:
                        entities[ent_name].append(" ".join(t))
                    else:
                        entities[ent_name] = [" ".join(t)]

            else:
                if token.ent_iob_ in ["O", ""]:
                    if self.word_vocab.word2index(token.text) == 0:
                        new_text_token_list.append(u"<unk>")
                        if u"<unk>" in entities:
                            entities[u"<unk>"].append(token.text)
                        else:
                            entities[u"<unk>"] = [token.text]
                    else:
                        new_text_token_list.append(token.text)

        return new_text_token_list, entities

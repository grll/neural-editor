import spacy
import logging


class GrllPreprocessor:
    """ Define all the preprocessing on sentence for the neural editor.

    Attributes:
        _word_vocab: a word vocab object which implement the word2index method with 0 being unknown index.
        _entities_to_replace: a list of entities to replace, other named entities will be ignored.
        _nlp: loaded spacy model used for NER (depend on the language)
    """

    def __init__(self, word_vocab=None, lang="de"):
        logging.info("Initialize preprocessor in `{}` with word_vocab: {}.".format(lang, word_vocab is None))
        self._word_vocab = word_vocab
        self._entities_to_replace = ["loc", "org", "per", "ordinal"]
        if lang == "de":
            logging.info("Load spacy model: `{}`.".format('de_core_news_sm'))
            self._nlp = spacy.load('de_core_news_sm')
            logging.info("Finished loading spacy model.")
        else:
            raise NotImplementedError

    def preprocess(self, sentence):
        """ Preprocess a given sentence.

        Args:
            sentence: the original sentence to preprocess.

        Returns:
            dict_entities, preprocessed_sentence
        """
        logging.debug("Preprocessing sentence: {}".format(sentence.encode("utf8")))
        doc = self._nlp(sentence)
        tokens_list, entities = self.replace_named_entities(doc)
        if self._word_vocab is not None:
            tokens_list, entities = self.replace_unknown_entities(tokens_list, entities)
        logging.debug("preprocessed_sentence: {}".format(" ".join(tokens_list).encode("utf8")))
        logging.debug("entities: {}".format(entities))
        return " ".join(tokens_list), entities

    def replace_named_entities(self, token_list):
        """ Replace the entities in the token list by their named entities and store the entities in a dict.

        Args:
            token_list: A list of spacy token or a doc.

        Returns:
            new_text_token_list, entities: a list of string token corresponding to the preprocessed sentence and a dict
                                           of entities.
        """
        logging.debug("Replacing named entities for token list: {}".format(token_list))
        new_text_token_list = []
        entities = {}
        for token in token_list:
            if token.ent_iob_ in ["O", ""]:
                new_text_token_list.append(token.text)
            elif token.ent_iob_ in ["B", "I"]:
                ent_name = token.ent_type_.lower()
                if ent_name in self._entities_to_replace:
                    if token.ent_iob_ == "B":
                        new_text_token_list.append("<" + ent_name + ">")
                        full_entity_txt = self.retrieve_full_entity_from_token(token)
                        if ent_name in entities:
                            entities[ent_name].append(full_entity_txt)
                        else:
                            entities[ent_name] = [full_entity_txt]
                else:
                    new_text_token_list.append(token.text)
            else:
                raise Exception("Failed to recognize if the token had an entity.")

        logging.debug("New token list: {}".format(new_text_token_list))
        logging.debug("Entities: {}".format(entities))
        return new_text_token_list, entities

    def retrieve_full_entity_from_token(self, token):
        """ Return the full entity as text from a beginning token

        Args:
            token: a token beginning an entity according to spacy

        Returns:
            full_entity_txt: A string representing the full entity.
        """
        logging.debug("Retrieving the full entity from token {}".format(token))
        if token.ent_iob_ != "B":
            raise Exception("The token is not the beginning of a spacy entity.")

        t = [token.text]
        for token in token.doc[token.i + 1:]:
            if token.ent_iob_ == "I":
                t.append(token.text)
            else:
                break
        logging.debug("Entity retrieved {}".format(" ".join(t).encode("utf8")))
        return " ".join(t)

    def replace_unknown_entities(self, tokens_list, entities):
        """ Replace all the unknown entities in the tokens_list and filled them up in the entities dict.

        Args:
            tokens_list: list of string tokens to potentially replace.
            entities: dict of entities to fill with the unkown entities.

        Returns:
            tokens_list: modified list of token to text with <unk>
            entities: modified dict with <unk> entities added
        """
        logging.debug("Replacing unknown entities in: \n{}\nwith original entities:\n{}".format(tokens_list, entities))
        if self._word_vocab is None:
            raise Exception("Can't replace unknown entities without a previously defined word_vocab")

        new_tokens_list = []
        new_entities = entities.copy()
        for token in tokens_list:
            if self._word_vocab.word2index(token) == 0:
                new_tokens_list.append(u"<unk>")
                if u"<unk>" in new_entities:
                    new_entities[u"<unk>"].append(token)
                else:
                    new_entities[u"<unk>"] = [token]
            else:
                new_tokens_list.append(token)
        logging.debug("New tokens list: \n{}\nnew entities:\n{}".format(new_tokens_list, new_entities))
        return new_tokens_list, new_entities

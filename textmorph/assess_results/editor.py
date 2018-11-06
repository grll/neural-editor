import torch
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.seq_batch import SequenceBatch
from itertools import izip
from gtd.utils import chunks
from textmorph.edit_model.editor import EditTrace
from textmorph.edit_model.encoder import EncoderOutput
import logging
from prettytable import PrettyTable

from sentence import Sentence
from results import GrllResults


class GrllNeuralEditor():
    """Perform modified edition function on an existing editor model."""

    def __init__(self, editor):
        """Initialise with an editor from the neural editor model."""
        logging.info("Initialize the modified Editor.")
        self.editor = editor

    def edit(self, examples, max_seq_length=35, beam_size=5, batch_size=1024):
        """Add one argument random_edit_vector wich enforce edition with a random vector."""
        logging.debug("Performing an edit on {} examples:\n {}".format(len(examples), examples))
        beam_list = []
        edit_traces = []
        for batch in chunks(examples, batch_size / beam_size):
            beams, traces = self._edit_batch(batch, max_seq_length, beam_size)
            beam_list.extend(beams)
            edit_traces.extend(traces)
        return beam_list, edit_traces

    # add sampling from random vector
    def _edit_batch(self, examples, max_seq_length, beam_size):
        """Add one argument random_edit_vector wich enforce edition with a random vector."""
        source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, _, edit_embed = self.editor._batch_editor_examples(
            examples)
        encoder_input = self.editor.encoder.preprocess(source_words, insert_words, insert_exact_words, delete_words,
                                                       delete_exact_words, edit_embed)
        encoder_output = self.encoder_generate_edits(encoder_input)

        beams, decoder_traces = self.editor.test_decoder_beam.decode(examples, encoder_output,
                                                                     weighted_value_estimators=[]
                                                                     , beam_size=beam_size, prefix_hints=[[]]
                                                                     , sibling_penalty=0, max_seq_length=max_seq_length)

        return beams, [EditTrace(ex, d_trace.beam_traces[-1]) for ex, d_trace in izip(examples, decoder_traces)]

    def encoder_generate_edits(self, encoder_input):
        """ Draw uniform random vectors with given norm, and use as edit vector """
        source_words = encoder_input.source_words
        source_word_embeds = self.editor.encoder.token_embedder.embed_seq_batch(source_words)
        insert_embeds = self.editor.encoder.token_embedder.embed_seq_batch(encoder_input.insert_words)
        delete_embeds = self.editor.encoder.token_embedder.embed_seq_batch(encoder_input.delete_words)

        insert_embeds_exact = self.editor.encoder.token_embedder.embed_seq_batch(encoder_input.insert_exact_words)
        delete_embeds_exact = self.editor.encoder.token_embedder.embed_seq_batch(encoder_input.delete_exact_words)

        source_encoder_output = self.editor.encoder.source_encoder(source_word_embeds.split())
        source_embeds_list = source_encoder_output.combined_states
        source_embeds = SequenceBatch.cat(source_embeds_list)
        # the final hidden states in both the forward and backward direction, concatenated
        source_embeds_final = torch.cat(source_encoder_output.final_states, 1)  # (batch_size, hidden_dim)

        edit_encoded = self.editor.encoder.edit_encoder(insert_embeds, insert_embeds_exact, delete_embeds,
                                                        delete_embeds_exact)

        # the random vector is computed as in rand_p_noise (see in edit_encoder)
        torch.manual_seed(7)
        batch_size, edit_dim = edit_encoded.size()
        rand_draw = GPUVariable(torch.randn(batch_size, edit_dim))
        rand_draw = rand_draw / torch.norm(rand_draw, p=2, dim=1).expand(batch_size, edit_dim)
        rand_norms = (torch.rand(batch_size, 1) * self.editor.encoder.edit_encoder.norm_max).expand(batch_size,
                                                                                                    edit_dim)
        edit_embed = rand_draw * GPUVariable(rand_norms)

        agenda = self.editor.encoder.agenda_maker(source_embeds_final, edit_embed)
        return EncoderOutput(source_embeds, insert_embeds_exact, delete_embeds_exact, agenda)

    def run_model(self, config_run_edit_model, dataloader, postprocessor):
        """ Run the editor model previously saved as attribute

        Args:
            config_run_edit_model: a dictionary corresponding to the current config_run edit_model
            dataloader: a data loader which generate the preprocessed data
            postprocessor:

        Returns:
            results an instance of the GrllResult class with the results from running the model
        """
        logging.info("Editing the sentences and generating results.")

        log_data = {  # A dictionary storing information about the model_run for logging purpose.
            "batches_len": [],
            "edit_traces_len": [],
            "filtered_candidates_len": [],
            "cum_results_len": []
        }
        results = GrllResults()  # a class to store the results from the model run.

        for idx, (preprocessed_sentence, entities, original_sentence) in enumerate(
                dataloader.generate_one_preprocessed_sample()):
            sentence = Sentence(preprocessed_sentence.split(" "),
                                entities=entities,
                                original_sentence=original_sentence,
                                preprocessed_sentence=preprocessed_sentence)

            batch = [sentence] * config_run_edit_model["random_edit_vector_number"]

            _, edit_traces = self.edit(batch)
            filtered_candidates = postprocessor.postprocess_filter(edit_traces,
                                                                   min_number_of_token=config_run_edit_model["min_number_of_token"])
            results.add_candidates(filtered_candidates)

            log_data["batches_len"].append(len(batch))
            log_data["edit_traces_len"].append(len(edit_traces))
            log_data["filtered_candidates_len"].append(len(filtered_candidates))
            log_data["cum_results_len"].append(len(results))

            if config_run_edit_model["max_iter"] is not None:
                if idx == (config_run_edit_model["max_iter"] - 1):
                    break

        x = PrettyTable()  # logging the log_data in a prettytable.
        x.field_names = ["batches_len", "edit_traces_len", "filtered_candidates_len", "cum_results_len"]
        for batch_len, edit_trace_len, filtered_candidate_len, cum_result_len in zip(log_data["batches_len"],
                                                                                     log_data["edit_traces_len"],
                                                                                     log_data[
                                                                                         "filtered_candidates_len"],
                                                                                     log_data["cum_results_len"]):
            x.add_row([batch_len, edit_trace_len, filtered_candidate_len, cum_result_len])
        logging.info("\n" + x.get_string())
        logging.info("{} filtered candidates were generated.".format(len(results)))

        return results

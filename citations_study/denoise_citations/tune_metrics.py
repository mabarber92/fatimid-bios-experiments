from main_helpers import initialiseEmbedModel, compute_embeddings, select_layer_clean, compute_cut
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

class testSetEvaluator():
    """Take a test set and evaluate the citation cutter for the specified range of parameters
    Can output csvs of performance for each sentence in the parameter space, plus average performance
    across the parameter space with a heatmap
    Can be used to """
    def __init__ (self, test_csv, similarity_threshold_range = np.arange(0.05, 0.5, 0.05), 
                  layers=[0, 5, 6, 7, 8, 9, 10, 11, 12], model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix",
                  test_col = "test_sequences", ground_truth_col = "ground_truth", run_tests = True, multi_process = True):
        
        # Check and set the parameters - we do this first in case these values have errors
        self._check_set_parameters(similarity_threshold_range, layers)

        # Prepare the test set
        self.load_test_set(test_csv, test_col, ground_truth_col)

        # Set the multiprocessing
        self.toggle_multi_process(multi_process)
        
        # Now we know we have valid parameters and data we can initiatise models
        # Initialise the embedding model
        self.tokenizer, self.transformer_model = initialiseEmbedModel(model_name)

        # If we're running tests when we initialise the evaluator then we run pipeline
        if run_tests:
            self.run_pipeline()

    def toggle_multi_process(self, activate):
        """Function to activate and deactivate multiprocessing for functions that require it"""
        self.multi_process = activate
        if self.multi_process:
            self.cpu_count = cpu_count() - 1
            print(f"Activated multi processing with {cpu_count() - 1} cores")
    
    def load_test_set(self, test_csv, test_col, ground_truth_col):
        """Load the test csv as dataframe and check that the data is ready to be processed"""
        self.test_df = pd.read_csv(test_csv)
        self.test_sequences = self.test_df[test_col].to_list()
        self.ground_truth = self.test_df[ground_truth_col].to_list()
        if len(self.test_sequences) != len(self.ground_truth):
            raise ValueError("Mismatch between test sequences and ground truth - check your data!")
        print(f"Loaded {len(self.test_sequences)} lines of test data")
    
    def _check_set_parameters(self, similarity_threshold_range, layers):
        """This function checks and sets the parameters for tuning. If the class is initialised,
        use this to change parameters - do not tweak the variables directly as it may cause value errors"""
        if isinstance(similarity_threshold_range, np.ndarray):
            self._similarity_thresholds = similarity_threshold_range.tolist()
        elif type(similarity_threshold_range) == list:
            self._similarity_thresholds = similarity_threshold_range[:]
        else:
            raise ValueError("A np arrange object or a list of values must be supplied for the similarity threshold")
        if type(layers) == list:
            self._layers = layers[:]
        else:
            raise ValueError("A list of layers must be supplied for the hidden layers")
        print(f"""{len(self._similarity_thresholds)} similarities to test\n {len(self._layers)} layers to test\n 
              Total number of tests: {len(self._similarity_thresholds)*len(self._layers)}""")


    def embed_input_sequences(self):
        """Compute embeddings and required tokens and offsets that can then be used for tests
        throughout the parameter space. Only needs to be run once per model and test set"""
        self.offsets = []
        self.tokens_as_strings = []
        self.embeddings = []
        for test_sequence in self.test_sequences:
            offsets, tokens_as_strings, embeddings = compute_embeddings(test_sequence, self.tokenizer, self.transformer_model)
            self.offsets.append(offsets)
            self.tokens_as_strings.append(tokens_as_strings)
            self.embeddings.append(embeddings)
    
  
    def run_parameter_set(self, threshold, hidden_layer):
        """Take a specified parameter set and use it to cut all of the sequences, using existing embeddings
        returns a list of cut sequences"""
        cut_sequences = []
        for test_sequence, offsets, tokens_as_strings, embeddings in zip(self.test_sequences, self.offsets, self.tokens_as_strings, self.embeddings):
            offsets, layer_embeddings = select_layer_clean(offsets, tokens_as_strings, embeddings, hidden_layer, self.tokenizer)
            cut_sequence = compute_cut(offsets, layer_embeddings, test_sequence, threshold)
            cut_sequences.append(cut_sequence)
        
        return cut_sequences

    def score_performance(self, cut_sequences):
        """Run through the outputs of one parameter run and check them against the ground_truth
        return a list of scores"""
        scores = []
        for cut_sequence, ground_truth_seq in zip(cut_sequences, self.ground_truth):
            score = len(ground_truth_seq.split()) - len(cut_sequence.split())
            scores.append(score)
        return scores
    
    def run_and_score(self, threshold, hidden_layer):
        """For a parameter set, run the cut and then score them.
         Return a list of lists that reflects the - separate function allows for easy parallisation"""
        cut_sequences = self.run_parameter_set(threshold, hidden_layer)
        scores = self.score_performance(cut_sequences)
        result = {"layer": hidden_layer, "threshold": threshold, "scores": scores}

        return result
    
    def record_scores(self, layer, threshold, scores, performance_matrix):
        """Take all the scores for one layer and threshold run - append to the test_df and add summary
        to performance matrix"""
        # Append list of scores to the main test_df
        self.test_df[f"{layer}_{threshold}"] = scores
        # Calculate sum and output
        total_score =sum(scores)
        performance_matrix.append({"layer": layer, "threshold": threshold, "score": total_score})
        return performance_matrix


    def score_test_parameters(self, full_csv = None):
        """Function that runs through the whole parameter space and tests - returns scored_df for parameter pairs
        and cumulative scores for each parameter set"""
        # Prepare the embeddings
        self.embed_input_sequences()

        # Initiate storage for overall performance - as a matrix
        performance_matrix = []

        # Loop through layers
        for layer in tqdm(self._layers):
            print(layer)

            # If parralelising, initiate pool and imap
            if self.multi_process:
                args = [(threshold, layer) for threshold in self._similarity_thresholds]
                with Pool(self.cpu_count) as pool:
                    all_scores = pool.starmap(self.run_and_score, args)
                
                for score in all_scores:
                    performance_matrix = self.record_scores(score["layer"], score["threshold"], score["scores"], performance_matrix)

            # If not loop and append to scores
            else:
                for threshold in self._similarity_thresholds:
                    scores = self.run_and_score(threshold, layer)
                    performance_matrix = self.record_scores(layer, threshold, scores["scores"], performance_matrix)

        # Transform matrix dict into dataframe and pivot
        matrix_df = pd.DataFrame(performance_matrix)
        matrix_df = matrix_df.pivot(index="layer", columns="threshold", values="score")

        # If full_csv path - output the full results
        if full_csv:
            matrix_df.to_csv(full_csv, encoding='utf-8-sig')

        # Fetch the best-scoring parameters
        # Get all those values 0 or less - as positive values are not good
        valid = matrix_df.stack()[lambda s: s <=0]
        
        # Find all parameters that have best score in that set
        best_all = valid[valid == valid.max()]
        print("Best parameters for the test set:")
        print(best_all)



    def run_pipeline(self):
        """Run all components of the pipeline in series"""
        self.score_test_parameters(full_csv="full_parameter_test.csv")
        self.test_df.to_csv("full_parameter_test_sentences.csv")
    
def prepare_eval_set(sequences_csv, out_csv, input_col, base_col="test_sequences", ground_truth_col = "ground_truth", sample_size = 100):
    """Take a csv of sentences, select a sample size at random, add a duplicate column named for evaluation
    and write out as a csv"""
    input_df = pd.read_csv(sequences_csv)

    # Randomly sample rows
    sample_df = input_df.sample(n=sample_size)

    # Add new column
    sample_df[ground_truth_col] = sample_df[input_col]
    sample_df = sample_df.rename(columns = {input_col : base_col})

    # Write out sample data
    sample_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    csv_in = "../../data/sira_citations.csv"
    csv_out = "../../data/citations_cut_evaluation.csv"

    # prepare_eval_set(csv_in, csv_out, "search_result")
    testSetEvaluator(csv_out)


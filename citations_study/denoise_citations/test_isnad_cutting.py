"""Take a file with labelled isnads - retrieve the isnads plus x following tokens - store as evaluation file and run through tune_metrics
This is a subtest for testing performance of using embedding arithmetic to identify the end of isnads and cut them"""

from tune_metrics import testSetEvaluator
import re
import pandas as pd

START_TAGS = ["@IB@", "@IBR@", "@IBC@", "@IBF@", "@IBFT@", "@IBT@", "@IBA@", "@IBET@", "@IET@", "@IBCT@"]
END_TAGS = ["@MB@", "@MBT@", "@IECT@", "@IEFT@"]

def create_test_set(tagged_text, csv_out, post_isnad_tokens = 20,  base_col="test_sequences", ground_truth_col = "ground_truth", sample_size = 20):
    """Take a text with defined tags - split out the isnads and create an evaluation set out of them.
    Extract all using the given metric, output to a df and then randomly sample to the given sample size
    Keep the tag in the evaluation data - to allow for filtering"""

    # Open the file
    with open(tagged_text, encoding='utf-8') as f:
        text = f.read()
    
    # Set regexes based on the tags
    isnad_splitter = "|".join(START_TAGS)
    matn_splitter = "|".join(END_TAGS)

    # Split first on the isnad splitter - loop through those splits and produce the test sequences and ground truth
    out_data = []
    splits = re.split(isnad_splitter, text)
    for split in splits:
        # Split on the end tag
        splits = re.split(matn_splitter, split)
        
        if len(splits) > 2:
            # Clean out the Latin text in both splits
            isnad = re.sub("[a-zA-Z\d@~#]+", "", splits[0])
            post_isnad = re.sub("[a-zA-Z\d@~#]+", "", splits[1])

            # Capture the set post_isnad_tokens from the second part of the split
            post_isnad = post_isnad.split()
            if len(post_isnad) > post_isnad_tokens:
                reduced = " ".join(post_isnad[:post_isnad_tokens])
            else:
                reduced = " ".join(post_isnad)

            test_sequence = " ".join([isnad, reduced])
            
            # Output into test data
            out_data.append({base_col: test_sequence, ground_truth_col: isnad})
    
    # Transform list of dicts into dataframe
    df = pd.DataFrame(out_data)

    # Print out some basic info
    print(f"{len(df)} total splits produced from the tagged text")
    if len(df) < sample_size:
        print("Data is smaller than specified sample size")

    # Take the sample
    df.sample(n=sample_size)

    # Output the csv
    df.to_csv(csv_out, encoding='utf-8-sig', index=False)


def fetch_all_tags(tagged_text):
    
    with open(tagged_text, encoding='utf-8') as f:
        text = f.read()
    
    all_tags = re.findall(r"@\w+@", text)

    unique_tags = list(set(all_tags))

    print(unique_tags)

if __name__ == "__main__":
    tagged_text = "isnad_test/Churis_0207Waqidi.Maghazi.Shamela0023680-ara1.txt"
    eval_data = "isnad_test/Churis_eval_data.csv"
    # fetch_all_tags(tagged_text)

    # create_test_set(tagged_text, eval_data)
    testSetEvaluator(eval_data, output_folder="isnad_test/camelbert-ca_cut_test_rolling_base", rolling_base_sim=True)


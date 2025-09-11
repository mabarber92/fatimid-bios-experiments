import re
from openiti.helper.funcs import read_text
from openiti.helper.ara import ar_tok_cnt
import pandas as pd
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

class mARkdownFile():
    """"Creates a python object from a openITI mARkdown file, which can then be parsed in
    various ways"""
    def __init__ (self, file_path, header_pattern = ["### ", r"\|", " "], bio_patterns= [r"### \$+ ", "#~:section:"], report=True):
        """Read the text into the object using a file"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        # Read in OpenITI text - split off header        
        self.mARkdown_text = read_text(file_path, remove_header=True)
        
        # Set the header pattern list - checking that its a valid list of parts
        if len(header_pattern) != 3:
            raise ValueError
        self.header_pattern = header_pattern
        self.bio_patterns = bio_patterns

        # Check kind of mARkdown annotation text has
        if report:
            print("Loaded text from:", file_path)
            print("Overall annotation statistics:")
            self.has_mARkdown()

    def has_mARkdown(self):
        """Produce a readout for the mARkdown - how many of each level"""
        # Produce regex and search for headers
        header_regex = self.build_header_tag()
        header_list = re.findall(header_regex, self.mARkdown_text)

        if len(header_list) == 0:
            print("No standard mARkdown headers in this text")
        else:
            unique_headers = list(set(header_list))
            unique_headers.sort(key=len)
            for header in unique_headers:
                print(f"Header: {header} - Count: {header_list.count(header)}")
        
        # Produce regexes for bio headers
        bio_header_regex = self.build_header_tag(bio_header=True)
        bio_header_list = re.findall(bio_header_regex, self.mARkdown_text)
        if len(bio_header_list) == 0:
            print("No bio headers in this text")
        else:
            unique_bio_headers = list(set(bio_header_list))
            unique_bio_headers.sort(key=len)
            for header in unique_bio_headers:
                print(f"Bio Header: {header} - Count: {bio_header_list.count(header)}")

    
    def count_heading_type(self, level, split = None):
        """Count the number of headings of a certain type - if split is given, use just the supplied
        text split to count the headings of that type - otherwise use whole text"""
        if split is not None:
            text_in = split
        else:
            text_in = self.mARkdown_text
        
        header_regex = self.build_header_tag(level)
        return len(re.findall(header_regex, text_in))
    
    def build_header_tag(self, level=None, bio_header=False):
         # If level is not specified build a regex that will capture all header levels
        if level is None:
            if bio_header:
                # If bio_header is True, use the bio header patterns
                tag = rf'(?:{self.bio_patterns[0]}|{self.bio_patterns[1]})'
            # Else follow standard header pattern
            else:
                tag = rf'{self.header_pattern[0]}{self.header_pattern[1]}+{self.header_pattern[2]}'
            return tag
        
        # If level not an int - return an error
        elif type(level) != int:
            raise ValueError
        
        # Else produce header for specified level
        else:
            if bio_header:
                # If bio_header is True, use the bio header patterns
                tag = self.bio_patterns[level - 1]
            else:
                # Use the standard header pattern
                tag = rf'{self.header_pattern[0]}{self.header_pattern[1]}{{{level}}}{self.header_pattern[2]}'
            return tag
    
    def get_level(self, text, bio_header=False):
        """Find the level of the header in the supplied text."""
        if not bio_header:
            level_piece = self.header_pattern[1]
            header_pieces = re.findall(rf'{level_piece}+', text)
            found_levels = [len(i) // len(level_piece) for i in header_pieces]
        else:
            # If bio_header is True, use the bio header patterns
            found_levels = []
            for i, pattern in enumerate(self.bio_patterns):
                print(text)
                print(pattern)
                header_pieces = re.findall(pattern, text)
                if len(header_pieces) > 0:
                    found_levels.append(i + 1)
        
        if len(found_levels) == 0:
            raise ValueError("Supplied text does not contain a valid header")
        elif len(found_levels) > 1:
            raise ValueError("Supplied text contains multiple headers")
        else:
            return found_levels[0]
        

    def header_contains_regex(self, regex, level=None, bio_header=False):
        """Return a regex that will match the header containing the regex"""
        header_regex = self.build_header_tag(level, bio_header=bio_header)
        return rf'({header_regex}.*?{regex}.*?\n)'

    def _check_and_choose_header(self, results_list):           
        if len(results_list) == 0:
            raise ValueError("No headers found containing the specified regex")
        elif len(results_list) > 1:
            print(f"{len(results_list)} headers found containing the regex")
            for i, result in enumerate(results_list):
                if type(result) == dict:
                    print(f"Result {i}: {result['header'][:1000]}...")
                else:
                    print(f"Result {i}: {result[:1000]}...")
            choice = int(input("Enter the number of the header to use: "))
        else:
            choice = 0
        return results_list[choice]



    def search_for_header(self, regex, level=None, bio_header=False):
        """Return the section of the text where the header contains the regex
        if level is none - search all levels and split based on retrieved level"""
        if level is None:
            full_regex = self.header_contains_regex(regex, level=None, bio_header=bio_header)
            results = re.findall(full_regex, self.mARkdown_text)
            header = self._check_and_choose_header(results)

            level = self.get_level(header, bio_header=bio_header)
        
        # If a level is not already specified, we take level where result is found and split the whole text on that level - so we can return the split
        splits = self.split_on_level(level, bio_header=bio_header)        
        matching_splits = []
        full_regex = self.header_contains_regex(regex, level=level, bio_header=bio_header)
        print(full_regex)
        for split in splits:
            if len(re.findall(full_regex, split["header"])) > 0:
                matching_splits.append(split)
        
        split = self._check_and_choose_header(matching_splits)
        return split 

    def split_on_level(self, level, split=None, get_length=False, bio_header=False):
        """Split the text by a level header. If a split is provided, use that text to split
        Return list of dicts {"header": "", "text" : ""}"""
        if split is not None:
            text_in = split
        else:
            text_in = self.mARkdown_text

        splits_out=[]        
        header_regex = self.build_header_tag(level, bio_header=bio_header)
        full_regex = rf"({header_regex}.*?\n)"
        print(full_regex)
        splits = re.split(full_regex, text_in)

        first_match = True
        first_match_idx = None

        for i, split in enumerate(splits):
            if re.match(header_regex, split):
                if first_match:
                    first_match_idx = i
                    first_match=False
                text = splits[i+1]                
                if get_length:
                    splits_out.append({"header": split, "text": text, "w_length": ar_tok_cnt(text)})
                else:
                    splits_out.append({"header": split, "text": text})
        if first_match_idx > 0:
            previous_matches = splits[0:first_match_idx]
            if type(previous_matches) == list:
                prev_text = " ".join(previous_matches)
            else:
                prev_text = previous_matches
            if get_length:
                prev_entry = [{"header": "Preceding text in split", "text": prev_text, "w_length": ar_tok_cnt(prev_text)}]
            else:
                prev_entry = [{"header": "Preceding text in split", "text": prev_text}]
            splits_out = prev_entry + splits_out
                
        return splits_out

    def heading_data_to_df(self, heading_splits, out_csv=None, drop_main_text=True):
        """Take a list of dicts for a heading split and convert to df
        export as csv if not None
        If drop_main_text then we do not retain the text underneath the headings (only the text of the heading itself) in the df
        or the export"""
        df = pd.DataFrame(heading_splits)
        df["header"] = df["header"].str.replace(r'\n+', "", regex=True)
        if drop_main_text:
            df = df.drop(columns= ["text"])
        if out_csv is not None:
            df.to_csv(out_csv, encoding='utf-8-sig', index=False)
        return df


    def get_len_subheadings(self, regex, level=None, bio_header=False, out_csv=None):
        """Find a heading with a specific regex - return a df with subheadings"""
        heading_split = self.search_for_header(regex, level, bio_header)
        if level is None:
            level = self.get_level(heading_split["header"], bio_header)
        subheading_level = level + 1
        subheading_splits = self.split_on_level(subheading_level, split=heading_split["text"], bio_header=bio_header, get_length=True)
        return self.heading_data_to_df(subheading_splits, out_csv=out_csv)
    
    def search_text(self, regex):
        """Search the whole text for a regex - return list of matches"""
        # Clean text of markdown for this kind of searching
        clean_text = re.sub(r'\n+', " ", self.mARkdown_text)
        clean_text = re.sub(r'[#~\$|:a-zA-Z\d=*()%"\'><@,.:;?؟،؛!«»\[\]-]', '', clean_text)
        clean_text = re.sub(r'\s+', " ", clean_text)
        results = re.findall(regex, clean_text)

        # Handle capture groups
        if len(results) > 0 and type(results[0]) == tuple:
            results = ["".join(result) for result in results]
        return results

class markdownCorpus():
    """Create a corpus of mARkdown files - can then be searched through"""
    def __init__ (self, corpus_path, meta_csv, start_date=0, end_date=1500,
                  book_uri_list = None, primary_only=True, multi_process=False):
        """Read the list of files into the object using a file"""

        if multi_process:
            self._activate_multi_process()
        else:
            self.multi_process = False
        
        # Prepare the metadata and file list        
        self.meta_df = pd.read_csv(meta_csv, sep="\t")
        self.filter_metadata(start_date=start_date, end_date=end_date, book_uri_list=book_uri_list, primary_only=primary_only)
        self.file_list = self._build_file_list(corpus_path)
        
        print(f"Loaded {len(self.file_list)} mARkdown files into corpus")
    
    def _activate_multi_process(self):
        """Activate multi processing for corpus searches"""
        self.multi_process = True
        self.cpu_count = cpu_count() - 1
        print(f"Activated multi processing with {cpu_count() - 1} cores")
    
    
    def filter_metadata(self, start_date=0, end_date=1500, book_uri_list = None, primary_only=True):
        """Filter the metadata dataframe based on the supplied criteria"""
        
        self.meta_df = self.meta_df[self.meta_df["date"].between(start_date, end_date)]
        if book_uri_list is not None:
            self.meta_df = self.meta_df[self.meta_df["book_uri"].isin(book_uri_list)]
        if primary_only:
            self.meta_df = self.meta_df[self.meta_df["status"] == "pri"]
        print(f"Filtered to {len(self.meta_df)} mARkdown files in corpus")

    def _build_file_list(self, corpus_path):
        """Build a list of files from the corpus path and meta csv"""
        self.meta_df["file_path"] = corpus_path + "/" + self.meta_df["local_path"].str.split("../", regex=False).str[-1]
        file_list = self.meta_df["file_path"].tolist()
        
        return file_list
    
    def _run_search_on_file(self, args):
        """Run a search on a single file - used for multi processing"""
        file, regex = args
        if os.path.exists(file):
            md_file = mARkdownFile(file, report=False)
            return {"file": file, "results": md_file.search_text(regex)}
        else:
            return {"file": file, "results": []}

    def search_corpus(self, regex, level=None, bio_header=False, out_csv=None, list_only=True):
        
        if self.multi_process:
            args = [(file, regex) for file in self.file_list]
            with Pool(self.cpu_count) as pool:
                results = tqdm(pool.imap(self._run_search_on_file, args), total=len(args))
                if list_only:
                    list_of_lists = [result["results"] for result in results]
                    return [item for sublist in list_of_lists for item in sublist]
                else:
                    return results

        else:
            results = []
            for file in tqdm(self.file_list):
                results.append(self._run_search_on_file((file, regex)))
            
            if list_only:
                list_of_lists = [result["results"] for result in results]
                return [item for sublist in list_of_lists for item in sublist]
            else:
                return results
        



"""Pipeline specific to al-Maqrizi's Muqaffa, where birth and death dates are given in the titles of
biographies. The script measures the length of the headings for each given period boundary. Results are
outputed to csv file and histograms of heading lengths produced (one for each period boundary)."""

from utilities.markdown_files import mARkdownFile
from tqdm import tqdm
import pandas as pd
import os
import plotly.express as px

DATE_PATTERN = r""
# Base patterns for locating dates in the Muqaffa - for reference
full_pattern = r"\[\d+(؟\s){0,4}\s?-\s(بعد|قبل)?\s?\d+\]"
birth_pattern = r"\[\d+(؟\s){0,4}\s?-\s\]"
death_pattern = r"\[-\s(بعد|قبل)?\s?\d+\]"

def regex_constructor(date, date_type="full"):
    """Construct a regex pattern that matches the given date in the biography heading.
    If death_date is True, the pattern will match death dates, otherwise birth dates."""
    # Need to ensure - for full date we use the date for the death date and capture anything before it
    # For 'death' and 'birth' we need to ensure the pattern only captures the incomplete entries
    if date_type == "full":
        regex = rf"\[\d+(؟\s){{0,4}}\s?-\s(بعد|قبل)?\s?{date}\]"
    elif date_type == "birth":
        regex = rf"\[{date}(؟\s){{0,4}}\s?-\s\]"
    elif date_type == "death":
        regex = rf"\[-\s(بعد|قبل)?\s?{date}\]"

    return regex

def summarise_date_headings(markdown_file, date, date_type):
    """Take one date, create the regex pattern search for relevant headings and return a list of section lengths and record the date type- for later processing"""
    regex = regex_constructor(date, date_type = date_type)

    # Get headings that match the regex at level 1 of the biographical headings
    matching_headings = markdown_file.search_for_header(regex, level=1, bio_header=True, get_length=True)
    # Loop through the headings and create a summary
    data = []
    for section in matching_headings:
        data.append({"header": section["header"], "w_length": section["w_length"], "date_type": date_type, "date": date})
    
    return data

def process_dates_list(markdown_file, dates_list, date_types = ["full", "birth", "death"]):
    """Loop through a list of dates, fetch and concatenate the data for that list of dates
    date_types specifies the patterns to use when constructing the date pattern - we go through the list in the 
    order in which it is given"""

    all_data = []

    for date in tqdm(dates_list):
        for date_type in date_types:
            data = summarise_date_headings(markdown_file, date, date_type = date_type)
            all_data.extend(data)
    
    return all_data

def analyse_by_label(section_df, out_dir, write_summary=True, lengths_below = None, lengths_above = None):
    
    if lengths_below:
        section_df = section_df[section_df["w_length"] < lengths_below]
    if lengths_above:
        section_df = section_df[section_df["w_length"] > lengths_above]

    # Create a df summarising data
    if write_summary:
        summary_data = []
        labels = section_df["label"].drop_duplicates().tolist()
        for label in labels:
            data = section_df[section_df["label"] == label]
            
            # Process summary data
            section_count = len(data)
            total_length = data["w_length"].sum()
            mean_length = data["w_length"].mean()
            median_length = data["w_length"].median()
            mode_length = data["w_length"].mode().iloc[0]
            max_length = data["w_length"].max()
            min_length = data["w_length"].min()
            summary_data.append({"label": label,
                                "section_count": section_count,
                                "total_length": total_length,
                                "mean_length": mean_length,
                                "median_length": median_length,
                                "mode_length": mode_length,
                                "max_length": max_length,
                                "min_length": min_length})
    
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(out_dir, "labels_summary.csv")
        summary_df.to_csv(csv_path, index=False)

    # Create a faceted histogram
    figure_path = os.path.join(out_dir, "histograms_by_label.html")
    fig = px.histogram(section_df, x="w_length", facet_col="label", facet_col_wrap=3, 
                       labels = {"w_length": "Biography Length"},
                       title = "The lengths of biographies in the Muqaffa by period")
    fig.write_html(figure_path)



def fetch_process_date_data(file_path, date_boundaries, out_dir):
    """Fetch the length of headings for each date boundary in the list - create a df with heading title, section length, category and date type
    perform summarisation on that df and create histograms - stored in ordered data directory (so we can try different date spans)
    date_boundaries = [{"label": "Fatimid", "date_range": [358, 568]}]"""
    
    # Check and create out_dir if needed
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Initialise the mARkdownFile object
    markdown_file = mARkdownFile(file_path, presplit_bio_level=1)

    # For each date list fetch the headings and the lengths of their corresponding sections
    # Return a dataframe of all heading titles and section lengths for later analysis
    df = pd.DataFrame()

    for date_entry in date_boundaries:
        date_range = date_entry["date_range"]
        dates_list = list(range(date_range[0], date_range[1]))

        results_list = process_dates_list(markdown_file, dates_list)
        results_df = pd.DataFrame(results_list)
        results_df["label"] = date_entry["label"]
        df = pd.concat([df, results_df])
    
    full_df_path = os.path.join(out_dir, "heading_lengths_by_label.csv")
    df.to_csv(full_df_path, index=False, encoding='utf-8-sig')
    
    # Process summaries
    analyse_by_label(df, out_dir)







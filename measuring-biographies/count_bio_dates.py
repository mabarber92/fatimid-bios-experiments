"""Pipeline specific to al-Maqrizi's Muqaffa, where birth and death dates are given in the titles of
biographies. The script measures the length of the headings for each given period boundary. Results are
outputed to csv file and histograms of heading lengths produced (one for each period boundary)."""

from utilities.marskdown_files import mARkdownFile

DATE_PATTERN = r""

def regex_constructor(date, date_type="full"):
    """Construct a regex pattern that matches the given date in the biography heading.
    If death_date is True, the pattern will match death dates, otherwise birth dates."""
    # Need to ensure - for full date we use the date for the death date and capture anything before it
    # For 'death' and 'birth' we need to ensure the pattern only captures the incomplete entries
    regex = ""
    return regex

def summarise_date_headings(markdown_file, date, date_type):
    """Take one date, create the regex pattern search for relevant headings and return the number of headings,
    , total length, average_section_length"""
    regex = regex_constructor(date, date_type = date_type)

    # Get headings that match the regex at level 1 of the biographical headings
    matching_headings = markdown_file.search_for_header(regex, level=1, bio_header=True, get_length=True)

    # Loop through the headings and create a summary
    section_lengths = []
    for section in matching_headings:
        section_lengths.append(section["w_length"])
    #CHECK AVERAGE SYNTAX
    return {"header_count": len(section_lengths), "total_length": section_lengths.sum(), "average_length": section_lengths.mean()}

def fetch_date_data(file_path, date_boundaries):
    """Fetch the length of headings for each date boundary in the list"""
    markdown_file = mARkdownFile(file_path)

    for date_entry in date_boundaries:
        date_range = date_boundaries[date_entry]["date_range"]
        dates_list = list(range(date_range))

        
        # For each date - constrct the regex and fetch the headings - then count lengths - possibly additional func to compute each different regex iteration

        for date in dates_list:
            date_summary = 




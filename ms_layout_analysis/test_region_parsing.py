"""This script should take an ALTO zip (without images) extract some basic information about the regions for the pages in that zip file.
It should do the following:
1. Loop through each file and extract the regions information, combining it into a  dictionary that summarises the data as follows:
    "region_name", "total_area", "total_token_count"
2. Loop through the dictionary to create a list of summary data
3. Loop through the resulting list of dictionaries and print out the summary data for each region type.
The goal is to teach you how loops function in Python, using a practical example. The majority of the code is written for you, but it
it contains some errors with variable names and loop usage. Your task is the fix the errors so that the script correctly prints the output"""

# === CODE Between these markers are functions used by the script - do not change these, there are no errors here! ===

import xml.etree.ElementTree as ET
import os

def parse_alto_regions(alto_xml_file):
    """Parse ALTO XML file to extract text regions.

    Args:
        alto_xml_file (str): Path to the ALTO XML file.

    Returns:
        list: List of dictionaries containing text region information.
        Dictionary structure as follows:
        {
            "Label": str,        # Region label (e.g., "Text", "Image", etc.)
            "Width": int,       # Width of the region
            "Height": int,      # Height of the region
            "Area": int,        # Area of the region (Width * Height)
            "TokenCount": int   # Number of tokens in the region
    """
    # Parse the ALTO XML file
    tree = ET.parse(alto_xml_file)
    root = tree.getroot()

    # Define the ALTO namespace
    ns_uri = root.tag[root.tag.find("{")+1 : root.tag.find("}")]
    ns = {"alto": ns_uri}

    # Fetch the region labels from OtherTag
    region_labels = {}
    for other_tag in root.findall(".//alto:OtherTag", ns):
        region_labels[other_tag.attrib["ID"]] = other_tag.attrib["LABEL"]


    regions = []
    for text_block in root.findall(".//alto:TextBlock", ns):
        
        # Get the region label using TAGREFS
        region_label = region_labels.get(text_block.attrib.get("TAGREFS"), "Unknown")

        # Get dimensions of region
        width = text_block.attrib.get("WIDTH")
        height = text_block.attrib.get("HEIGHT")
        area = int(width) * int(height) if width and height else None

        # Get the token count for the region
        full_text = []
        for string in text_block.findall(".//alto:String", ns):
            full_text.append(string.attrib.get("CONTENT", ""))
        token_count = len(" ".join(full_text).split())

        region_info = {
            "Label": region_label,
            "Width": width,
            "Height": height,
            "Area": area,
            "TokenCount": token_count
        }
       
        regions.append(region_info)

    return regions

# == End of code you should not change ==

# Path to the directory containing ALTO XML files - Change this to the path where you extracted your own data
alto_directory = "sample_data/muqaffa_or14533_alto"

# os.listdir() is used to create a list of all of the files in alto_directory
alto_files = os.listdir(alto_directory)

# 1. We loop through each file in the directory and extract the regions information using the given function
regions_summary = {}
for file in alto_files:
    
    # Set the file path
    file_path = os.path.join(alto_directory, file)

    # Get the regions information using the provided function
    regions = parse_alto_regions(file_path)
    
    # Loop through the regions to update the summary dictionary - This code functions as expected - do not change it!
    for region in regions:
        label = region["Label"]
        area = region["Area"]
        token_count = region["TokenCount"]

        if label not in regions_summary:
            regions_summary[label] = {"total_area": 0, "total_token_count": 0, "count": 0}
        
        regions_summary[label]["total_area"] += area
        regions_summary[label]["total_token_count"] += token_count
        regions_summary[label]["count"] += 1


# 2. Now convert the summary dictionary into a list of dictionaries for easier processing later
summary_list = []
for region in regions_summary:
    data = regions_summary[region]
    summary_list.append({
        "region_name": region,
        "total_area": data["total_area"],
        "total_token_count": data["total_token_count"],
        "count": data["count"]
    })

# 3. Finally, loop through the list and print out each region's summary data
total_regions = 0
region_types = 0
for summary in summary_list:

    # Get the data we need to print
    region_name = summary["region_name"]
    total_area = summary["total_area"]
    total_token_count = summary["total_token_count"]
    count = summary["count"]

    # Print the data nicely

    print("Region Summary:", region_name)
    print("Total Area:", total_area)
    print("Total Token Count:", total_token_count)
    print("Count:", count)
    print("----\n")
    
    total_regions += summary["count"]
    region_types += 1


print(total_regions, "regions found in total across all files.")
print(region_types, "different region types found.")

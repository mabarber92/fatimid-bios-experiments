"""Functions for the parsing of XML ontology files."""
import xml.etree.ElementTree as ET
import zipfile
import os

def open_xml_zip(zip_file, folder_destination=None):
    """Open and extract an XML ontology file from a ZIP archive.

    Args:
        zip_file (str): Path to the ZIP file containing the XML ontology.
        folder_destination (str, optional): Destination folder to extract the contents. Defaults to None.

    Returns:
        str: Path to the extracted XML file.
    """


    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        if folder_destination is None:
            folder_destination = zip_file.split(".zip")[0]
            
        zip_ref.extractall(folder_destination)

    return folder_destination

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



if __name__ == "__main__":
    # zip_path = "sample_data/muqaffa_or14533_alto.zip"
    # extracted_folder = open_xml_zip(zip_path)
    extracted_folder = "sample_data/muqaffa_or14533_alto"
    first_xml = os.listdir(extracted_folder)[2]
    xml_path = os.path.join(extracted_folder, first_xml)
    regions = parse_alto_regions(xml_path)
    print(regions)

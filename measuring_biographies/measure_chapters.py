from utilities.markdown_files import mARkdownFile


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parse mARkdown files.')
    parser.add_argument('file_dir', type=str, help='Path to the mARkdown file')
    args = parser.parse_args()
    
    # Create an instance of mARkdownFile and check for headers
    md_file = mARkdownFile(args.file_dir)

    regex = "اليازوري"
    md_file.get_len_subheadings(regex, bio_header=True, out_csv="yazuri_headings.csv")    
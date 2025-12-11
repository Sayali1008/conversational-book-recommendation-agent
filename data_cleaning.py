from utilities import safe_read_csv
import constants as const
import html
import pandas as pd
import re

# ----------------------
# Helper methods
# ----------------------
def normalize_author(author_col):
    # Ensure author column is string
    authors_list = author_col.fillna("").astype(str)
    
    # Replace separators in bulk
    authors_list = (authors_list
                    .str.replace(";", ",", regex=False)
                    .str.replace("&", ",", regex=False)
                    .str.replace(r"\s+and\s+", ",", regex=True)
                )
    
    # Remove editor markers in bulk
    authors_list = (authors_list
                    .str.replace(r"\(.*?editor.*?\)", "", regex=True, case=False)
                    .str.replace(r"\beditor\b", "", regex=True, case=False)
                    .str.replace(r"\bed\.\b", "", regex=True, case=False)
                    )

    # Collapse extra spaces
    authors_list = authors_list.str.replace(r"\s+", " ", regex=True).str.strip()

    # Split into lists
    authors_list = authors_list.str.split(",")

    # Trim whitespace inside list items
    authors_list = authors_list.apply(lambda lst: [item.strip() for item in lst if item.strip()])

    # Remove invalid values
    invalid_authors = ["unknown"]
    authors_list = authors_list.apply(lambda lst: [item for item in lst if item.lower() not in invalid_authors])

    # Deduplicate + title case + sort
    authors_list = authors_list.apply(lambda lst: sorted(set([item.title() for item in lst if isinstance(lst, list)])))

    return authors_list

def normalize_genre(genre_col):
    # Ensure string type
    genre_list = genre_col.fillna("").astype(str)

    # Replace common separators with commas
    for sep in const.COMMON_GENRE_DELIMS:
        genre_list = genre_list.str.replace(sep, ",", regex=False)
    
    # Collapse extra spaces
    genre_list = genre_list.str.replace(r"\s+", " ", regex=True).str.strip()

    # Split into lists
    genre_list = genre_list.str.split(",")

    # Trim whitespace inside list items
    genre_list = genre_list.apply(lambda lst: [item.strip() for item in lst if item.strip()])

    # Deduplicate + title case + sort
    # Remove truncated / obviously invalid entries
    genre_list = genre_list.apply(lambda lst: sorted(set([item.title() for item in lst if isinstance(lst, list) and len(item) > 2 and "..." not in item])))

    return genre_list

def normalize_text_field(text_col):
    # Ensure string type
    text_col = text_col.fillna("").astype(str)

    # Remove HTML tags
    text_col = text_col.str.replace(r"<[^>]+>", "", regex=True)

    # Remove escaped characters
    text_col = text_col.str.replace(r"[\n\t\r]", " ", regex=True)
    
    # Collapse multiple spaces
    text_col = text_col.str.strip().str.replace(r"\s+", " ", regex=True)
    
    # Decode HTML entities
    text_col = text_col.apply(html.unescape)
    
    # Remove control/non-printable characters
    text_col = text_col.apply(lambda s: "".join(ch for ch in s if ch.isprintable()))

    return text_col

# Handles duplicates based on identical title and author(s) combinations
def handle_duplicate_rows(df):
    # Make unique key using title and author
    title_lower = df["title"].str.lower()
    author_list_lower = df["author"].apply(lambda lst: tuple(sorted([item.lower() for item in lst])))
    df["title_author_key"] = list(zip(title_lower, author_list_lower))

    # Union genres for each title + author(s) group
    genre_union_map = {}
    for idx, row in df.iterrows():
        key = row["title_author_key"]
        if key not in genre_union_map:
            genre_union_map[key] = set()
        genre_union_map[key].update(row["genre"])

    # Convert sets to lists
    for key in genre_union_map:
        genre_union_map[key] = list(genre_union_map[key])

    # Apply the unioned genres to all rows
    df["genre_union"] = df["title_author_key"].map(genre_union_map)

    # Drop helper column
    df = df.drop(columns="title_author_key")

    return df

def run_pipeline(input_path, output_path):
    # Step 0: snapshot raw CSV
    # snapshot = snapshot_raw(input_csv, CONFIG["snapshot_dir"])

    # Step 1: schema validation
    df = safe_read_csv(input_path)
    
    # Ensure expected columns exist
    missing_cols = [c for c in const.INPUT_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input CSV: {missing_cols}")
    
    # Only use the needed columns for processing
    df = df[const.INPUT_COLS]
    
    # Step 2: keep raw backup columns
    for c in const.INPUT_COLS:
        df[c] = df[c].fillna("").astype(str)
        df[f"raw_{c}"] = df[c]  # keep original
    
    # Step 3: convert and standardize pages as integers
    df["pages"] = pd.to_numeric(df["pages"], errors="coerce").fillna(0).astype(int).clip(lower=0)

    # Step 4: normalize authors
    df["author"] = normalize_author(df["author"])

    # Step 5: normalize genres
    df["genre"] = normalize_genre(df["genre"])

    # Step 6: normalize titles and descriptions
    df["title"] = normalize_text_field(df["title"])
    df["desc"] = normalize_text_field(df["desc"])

    # Step 7: handling duplicate data
    df = handle_duplicate_rows(df)

    # Step 8: Remove missing or incomplete data
    # Drop rows only if all essential fields are invalid:
    # missing authors AND genres AND descriptions AND (a blank OR "unknown" title).
    mask = (df["author"].apply(lambda x: len(x) == 0) &
            df["desc"].apply(lambda x: (not isinstance(x, str)) or (x.strip() == "")) &
            df["genre_union"].apply(lambda x: len(x) == 0) &
            df["title"].apply(lambda t: (not isinstance(t, str)) or ("unknown" in t.lower())))
    
    num_rows_removed = mask.sum()    
    print(f"Number of rows dropped when any essential fields are found invalid: {num_rows_removed}")
    
    # Step 9: Save the cleaned data
    df_cleaned = df[~mask].copy()
    df_cleaned.to_csv(output_path, index=False, header=True, columns=const.OUTPUT_COLS)

def main():
    input_path = "data/GoodReads_100k_books.csv"
    output_path = "data/books_cleaned.csv"
    run_pipeline(input_path, output_path)

if __name__ == "__main__":
    main()
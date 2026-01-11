import ast
import html

import pandas as pd
from pydantic import ConfigDict, validate_call

from config import *


# ----------------------
# Helper methods
# ----------------------
def parse_list_value(val):
    """
    Parse a string value into a list of cleaned strings.
    Args: val (str or NaN): The input string representing a list or comma-separated values.
    Returns: list[str]: A list of stripped strings, empty if input is NaN, empty, or cannot be parsed.
    """
    if pd.isna(val) or val.strip() == "":
        return []
    val = val.strip()

    # Try to parse as Python list
    if val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)
            return [c.strip() for c in parsed if isinstance(c, str)]
        except Exception:
            return []

    # Otherwise, comma-separated
    for sep in COMMON_DELIMS:
        val = val.replace(sep, ",")

    return [c.strip() for c in val.split(",") if c.strip()]


def normalize_author_column(author_col):
    """
    Normalize and clean a DataFrame column containing author names.
    Args: author_col (pd.Series): Column of raw author strings.
    Returns: pd.Series: Column where each row is a sorted list of valid author names in title case.
    """
    # basic cleanup before parsing
    cleaned_col = (
        author_col.fillna("")
        .astype(str)
        .str.replace(";", ",", regex=False)
        .str.replace("&", ",", regex=False)
        .str.replace(r"\s+and\s+", ",", regex=True)
        .str.replace(r"\(.*?editor.*?\)", "", regex=True, case=False)
        .str.replace(r"\beditor\b", "", regex=True, case=False)
        .str.replace(r"\bed\.\b", "", regex=True, case=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # parse each row value in the column
    cleaned_col = cleaned_col.apply(parse_list_value)

    # convert valid authors to title case and remove invalid ones
    invalid_authors = {"unknown"}
    cleaned_col = cleaned_col.apply(
        lambda author_row: sorted(
            {item.title() for item in author_row if item.lower() not in invalid_authors}
        )
    )

    return cleaned_col


def normalize_genre_column(genre_col):
    """
    Normalize and clean a column of genres, converting to lowercase, trimming, removing short/invalid entries, and sorting.
    Args: genre_col (pd.Series): Column of genre strings.
    Returns: pd.Series: Column where each row is a sorted list of valid genres.
    """
    cleaned_col = (
        genre_col.fillna("")
        .astype(str)
        .str.strip()
        .str.replace("&", "and", regex=False)
    )

    # parse each row value in the column
    cleaned_col = cleaned_col.apply(parse_list_value)

    # lowercase, trim, remove very short/invalid entries
    cleaned_col = cleaned_col.apply(
        lambda lst: [g.lower().strip() for g in lst if len(g) > 2 and "..." not in g]
    )

    # sort
    cleaned_col = cleaned_col.apply(lambda lst: sorted(set(lst)))

    return cleaned_col


def map_genres_to_top_or_other(genres_row, top_genres):
    """
    Map genres to a top-N list, replacing non-top genres with 'other'.
    Args:
        genres_row (list[str]): List of genres for a book.
        top_genres (set[str]): Set of top-N genres.
    Returns: list[str]: Sorted list with genres mapped to top or 'other'.
    """
    # any genre not in top_genres → "other"
    genre_list = set()
    for g in genres_row:
        if g in top_genres:
            genre_list.add(g)
        else:
            genre_list.add("other")

    return sorted(genre_list)


def reduce_to_top_genres(genre_col):
    """
    Reduce genres in a column to top-N most common genres, mapping all other genres to 'other'.
    Args: genre_col (pd.Series): Column of lists of genres.
    Returns: pd.Series: Column of lists with top-N genres or 'other'.
    """
    all_genres = genre_col.explode()
    top_genres = set(all_genres.value_counts().head(TOP_N_GENRES).index)
    return genre_col.apply(map_genres_to_top_or_other, args=(top_genres,))


def normalize_text_field(cleaned_col):
    """
    Clean a text column by removing HTML tags, unescaping HTML entities, removing escaped characters and control characters, and collapsing whitespace.
    Args: cleaned_col (pd.Series): Column of strings.
    Returns: pd.Series: Cleaned string column.
    """
    cleaned_col = cleaned_col.fillna("").astype(str)
    cleaned_col = cleaned_col.str.replace(r"<[^>]+>", "", regex=True)
    cleaned_col = cleaned_col.apply(html.unescape)
    cleaned_col = cleaned_col.str.replace(r"[\n\t\r]", " ", regex=True)
    cleaned_col = cleaned_col.str.strip().str.replace(r"\s+", " ", regex=True)
    cleaned_col = cleaned_col.apply(
        lambda s: "".join(ch for ch in s if ch.isprintable())
    )

    return cleaned_col


def keep_usable_books(df):
    """
    Filter books to retain only those with complete and valid metadata: title, at least one author, at least one genre, and description length >= MIN_DESC_LENGTH.
    Args: df (pd.DataFrame): DataFrame containing book metadata.
    Returns: tuple: (filtered DataFrame of usable books, fraction of books that are usable)
    """
    mask = (
        (df["title"].notna() & df["title"].str.strip() != "")
        & (df["description"].str.strip().str.len() >= MIN_DESC_LENGTH)
        & (df["authors"].notna() & df["authors"].apply(lambda lst: len(lst) > 0))
        & (df["genres"].apply(lambda lst: len(lst) > 0))
    )

    return df[mask].copy(), mask.mean()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def clean_books_data(logger, books_df) -> pd.DataFrame:
    """
    Clean and normalize the books DataFrame including authors, genres, titles, and descriptions.
    Args: books_df (pd.DataFrame): Raw books DataFrame.
    Returns: pd.DataFrame: Cleaned catalog of usable books.
    """
    books_df["authors"] = normalize_author_column(books_df["authors"])
    books_df["genres"] = reduce_to_top_genres(
        normalize_genre_column(books_df["categories"])
    )
    books_df["title"] = normalize_text_field(books_df["title"])
    books_df["description"] = normalize_text_field(books_df["description"])

    # A usable book must have: title, at least one author, description length ≥ X, at least one category
    books_df, usable_ratio = keep_usable_books(books_df)
    logger.info(f"Usable books: {usable_ratio:.2%}")

    # NOTE: No title + author duplicates were found in this dataset

    # Create a unique sequential integer starting from 0 for each book
    books_df["book_id"] = range(1, len(books_df) + 1)

    return books_df


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def clean_ratings_data(logger, ratings_df, cleaned_books_df) -> pd.DataFrame:
    """Clean ratings DataFrame, normalize titles, convert timestamps, deduplicate, transform scores to confidence, and filter by user/item thresholds."""

    # Drop rows with missing title, user_id, review/score
    ratings_df = ratings_df[ratings_df["title"].notna()]
    ratings_df = ratings_df[ratings_df["user_id"].notna()]
    ratings_df = ratings_df[ratings_df["review/score"].notna()]

    # Normalize titles with same techniques as for books data
    ratings_df["title"] = normalize_text_field(ratings_df["title"])
    ratings_df = pd.merge(
        ratings_df, cleaned_books_df[["book_id", "title"]], on="title", how="inner"
    )
    logger.info(f"ratings_df.columns: {ratings_df.columns.tolist()}")

    # Parse review/time from epoch time to datetime UTC
    ratings_df["datetime"] = pd.to_datetime(
        pd.to_numeric(ratings_df["review/time"]), unit="s", utc=True
    )

    # Deduplicate (user, book) pairs, group by (user_id, title) and keep one with most recent review/time
    ratings_df = ratings_df.sort_values("review/time").drop_duplicates(
        subset=["user_id", "title"], keep="last"
    )

    # Transform 1-5 ratings into confidence weights: scores ≤3 become 0, 4 becomes 1, and 5 becomes 2
    ratings_df["review/score"] = pd.to_numeric(
        ratings_df["review/score"], errors="coerce"
    )
    ratings_df["confidence"] = ratings_df["review/score"].clip(lower=3) - 3

    # Filter out interactions with zero confidence (ratings ≤ 3)
    logger.info(f"Ratings data size before filtering confidence=0: {len(ratings_df):,}")
    len_confidence_0 = len(ratings_df[ratings_df["confidence"] == 0])
    ratings_df = ratings_df[ratings_df["confidence"] > 0].copy()

    logger.info(f"Ratings data size after filtering confidence=0: {len(ratings_df):,}")
    logger.info(f"Removed {len_confidence_0:,} zero-confidence rows")
    return ratings_df

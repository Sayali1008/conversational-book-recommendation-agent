### Understanding the schema

| Query | Count/Percentage |
|---|---:|
| How many rows does the dataset have? | 100000 |

### Checking for missing or incomplete values

| Query | Count/Percentage |
|---|---:|
| How many books have missing titles? | 1 |
| How many books have no authors or author is blank or Unknown? | 15 |
| How many books have missing descriptions? | 6772 or 6.77% |
| How many books have missing genres? | 10467 or 10.47% |
| How many books have 0 pages? | 7752 |
| How many books have no ISBN code? | 14482 or 14.48% |
| How many books have no link? | None baabbyy |

### Look for anomalies or inconsistencies

| Query | Count/Percentage |
|---|---:|
| Number of unique genres? | 1182 |
| Are there any completely duplicated rows? | None baabbyy |
| Partial duplicates (titles + authors) | 92 |
| How many descriptions have HTML tags? | 14 |
| How many descriptions have escaped characters (`\t` or `\r` or ..)? | 24 |
| How many descriptions have HTML escaped characters (`&amp;` or `&#39;` or ..)? | 56 |
| How many descriptions have control characters (non-printable unicode) | 1881 |

### Data preprocessing pipeline

- Genre: Create a new list column by splitting comma-separated string genre into unique genres per row
- Duplicates: For partial duplicates (title + author), if there are two different descriptions or ISBNs, keep the two rows and union the genres, apply the unioned genres to both the rows so both rows get all the possible genres
- Descriptions: Normalize and handle HTML tags, escaped characters, HTML escaped characters, control characters (non-printable unicode)
- Authors: Normalize and handle HTML tags, escaped characters, HTML escaped characters, control characters (non-printable unicode)
- Titles: Normalize and handle HTML tags, escaped characters, HTML escaped characters, control characters (non-printable unicode)
- [TODO] Entries in genre, title, author, etc. which look like this are not normalized yet: Aurã©Lie Neyret, Bande DessinÃ©e, etc.

### Steps involved

- Titles, authors, and descriptions are normalized.
- Genre lists are cleaned and unioned for duplicates.
- Partial duplicates are handled without losing information.
- Rows that are completely useless (no authors, no genres, no description, unknown title) have been removed.
- The dataset is safe for embedding generation because all the key fields are now consistent and clean.

### Notes

**Coding Notes**

- .apply() is slow > use panda's built in vectorized operations instead
- .astype('category') > categorical fields
- pdf.read_csv(, usecols=) to load only required columns resulting in less memory usage


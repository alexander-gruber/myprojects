import re
from pybtex.database import parse_file
from collections import defaultdict

# ------------------------------------------------------------
# Authors
# ------------------------------------------------------------

def get_author_string(entry):
    """Return a normalized author string for grouping."""

    authors = entry.persons.get("author", [])

    return " ".join(
        " ".join(author.last_names)
        for author in authors
    ).lower()


def assign_year_suffixes(bib, cited_keys):
    """
    Assign a, b, c, ... to references having the same author(s)
    and publication year.

    Returns:
        dict: {bibtex_key: suffix}
    """

    groups = defaultdict(list)

    # Group references by author + year
    for key in cited_keys:

        if key not in bib.entries:
            continue

        entry = bib.entries[key]

        author = get_author_string(entry)
        year = entry.fields.get("year", "n.d.")

        groups[(author, year)].append(key)

    suffixes = {}

    for (author, year), keys in groups.items():

        if len(keys) == 1:
            suffixes[keys[0]] = ""

            continue

        # Sort according to bibliography order
        keys.sort(
            key=lambda key: (
                clean_title(bib.entries[key].fields.get("title", ""))
                .lower()
            )
        )

        for i, key in enumerate(keys):
            suffixes[key] = chr(ord("a") + i)

    return suffixes

def format_authors(authors):
    """Format BibTeX authors approximately according to APA 7."""

    formatted = []

    for author in authors:
        # BibTeX handles names such as:
        # Smith, John
        # John Smith
        # van der Waals, Johannes Diderik
        first = " ".join(author.first_names)
        middle = " ".join(author.middle_names)
        last = " ".join(author.last_names)
        prefix = " ".join(author.prelast_names)

        initials = []

        for name in author.first_names + author.middle_names:
            if name:
                initials.append(name[0] + ".")

        initials = " ".join(initials)

        if prefix:
            surname = f"{prefix} {last}"
        else:
            surname = last

        if initials:
            formatted.append(f"{surname}, {initials}")
        else:
            formatted.append(surname)

    if not formatted:
        return ""

    # APA 7:
    # 1 author
    # 2 authors -> &
    # 3–20 authors -> commas + &
    # >20 authors -> first 19 ... final author

    if len(formatted) == 1:
        return formatted[0]

    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"

    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    # APA rule for 21+ authors
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

def clean_title(title):
    """Remove common BibTeX braces used for capitalization protection."""

    title = title.replace("{", "").replace("}", "")
    title = re.sub(r"\s+", " ", title).strip()

    return title


# ------------------------------------------------------------
# DOI
# ------------------------------------------------------------

def format_doi(doi):
    if not doi:
        return ""

    doi = doi.strip()

    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)

    return f"https://doi.org/{doi}"


# ------------------------------------------------------------
# Individual reference
# ------------------------------------------------------------

def format_reference(entry):
    """Format one BibTeX entry approximately in APA 7."""

    entry_type = entry.type.lower()
    fields = entry.fields
    persons = entry.persons

    # Authors
    authors = format_authors(persons.get("author", []))

    # Year
    year = fields.get("year", "n.d.")

    # Title
    title = clean_title(fields.get("title", ""))

    # --------------------------------------------------------
    # Journal article
    # --------------------------------------------------------

    if entry_type == "article":

        journal = fields.get("journal", "")
        volume = fields.get("volume", "")
        number = fields.get("number", fields.get("issue", ""))
        pages = fields.get("pages", "")

        result = f"{authors} ({year}). {title}. {journal}"

        if volume:
            result += f", {volume}"

        if number:
            result += f"({number})"

        if pages:
            pages = pages.replace("--", "–")
            result += f", {pages}"

        result += "."

    # --------------------------------------------------------
    # Book
    # --------------------------------------------------------

    elif entry_type == "book":

        publisher = fields.get("publisher", "")

        result = f"{authors} ({year}). {title}."

        if publisher:
            result += f" {publisher}"

    # --------------------------------------------------------
    # Book chapter
    # --------------------------------------------------------

    elif entry_type == "incollection":

        editor = format_authors(persons.get("editor", []))
        booktitle = clean_title(fields.get("booktitle", ""))
        publisher = fields.get("publisher", "")
        pages = fields.get("pages", "")

        result = f"{authors} ({year}). {title}. "

        if editor:
            result += f"In {editor} (Ed.), "

        result += booktitle

        if pages:
            pages = pages.replace("--", "–")
            result += f" (pp. {pages})"

        if publisher:
            result += f". {publisher}"

        result += "."

    # --------------------------------------------------------
    # Conference paper
    # --------------------------------------------------------

    elif entry_type == "inproceedings":

        booktitle = clean_title(fields.get("booktitle", ""))
        publisher = fields.get("publisher", "")
        pages = fields.get("pages", "")

        result = f"{authors} ({year}). {title}. In {booktitle}"

        if pages:
            pages = pages.replace("--", "–")
            result += f" (pp. {pages})"

        if publisher:
            result += f". {publisher}"

        result += "."

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    else:

        result = f"{authors} ({year}). {title}."

        journal = fields.get("journal")
        publisher = fields.get("publisher")

        if journal:
            result += f" {journal}."

        elif publisher:
            result += f" {publisher}."

    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    doi = format_doi(fields.get("doi"))

    if doi:
        result += f" {doi}"

    return result


# ------------------------------------------------------------
# Citation extraction
# ------------------------------------------------------------

def get_cited_keys(tex):
    """
    Find all \\citep{...} commands.

    Also handles:
        \\citep[see][p. 12]{smith2020}
        \\citep{smith2020,jones2022}
    """

    pattern = r"\\citep(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}"

    keys = []

    for match in re.finditer(pattern, tex):

        for key in match.group(1).split(","):
            key = key.strip()

            if key:
                keys.append(key)

    # Remove duplicates
    return list(dict.fromkeys(keys))


# ------------------------------------------------------------
# Replace citations
# ------------------------------------------------------------

def format_intext_citation(keys, bib, year_suffixes):

    citations = []

    for key in keys:

        if key not in bib.entries:
            citations.append(f"[MISSING: {key}]")
            continue

        entry = bib.entries[key]

        authors = entry.persons.get("author", [])

        if len(authors) == 0:
            author = "Anonymous"

        elif len(authors) == 1:
            author = " ".join(authors[0].last_names)

        elif len(authors) == 2:
            author = (
                " ".join(authors[0].last_names)
                + " & "
                + " ".join(authors[1].last_names)
            )

        else:
            author = " ".join(authors[0].last_names) + " et al."

        year = entry.fields.get("year", "n.d.")

        # Add a/b/c suffix
        suffix = year_suffixes.get(key, "")

        citations.append(
            (author, year, suffix)
        )

    # APA ordering of multiple parenthetical citations
    citations.sort(
        key=lambda x: (
            x[0].lower(),
            x[1],
            x[2]
        )
    )

    return "(" + "; ".join(
        f"{author}, {year}{suffix}"
        for author, year, suffix in citations
    ) + ")"

def replace_citations(tex, bib, year_suffixes):

    pattern = r"\\citep(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}"

    def replacement(match):

        keys = [
            key.strip()
            for key in match.group(1).split(",")
        ]

        return format_intext_citation(
            keys,
            bib,
            year_suffixes
        )

    return re.sub(pattern, replacement, tex)


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

# def latex_to_plaintext(tex, bib_file):
def latex_to_plaintext(tex, bib):

    # bib = parse_file(bib_file)

    # Find citations
    cited_keys = get_cited_keys(tex)

    # --------------------------------------------------------
    # Assign a/b/c suffixes
    # --------------------------------------------------------

    year_suffixes = assign_year_suffixes(
        bib,
        cited_keys
    )

    # --------------------------------------------------------
    # Replace citations
    # --------------------------------------------------------

    text = replace_citations(
        tex,
        bib,
        year_suffixes
    )

    # --------------------------------------------------------
    # Create bibliography
    # --------------------------------------------------------

    references = []

    for key in cited_keys:

        if key not in bib.entries:
            print(
                f"WARNING: {key} not found in bibliography"
            )
            continue

        entry = bib.entries[key]

        reference = format_reference(entry)

        # Add year suffix to bibliography
        year = entry.fields.get("year", "n.d.")
        suffix = year_suffixes.get(key, "")

        reference = reference.replace(
            f"({year}).",
            f"({year}{suffix}).",
            1
        )

        references.append(reference)

    # APA alphabetical ordering
    references.sort(key=lambda x: x.lower())

    bibliography = "\n\n".join(references)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    output = (
        text.rstrip()
        + "\n\n"
        + "References:\n\n"
        + bibliography
        + "\n"
    )

    return output

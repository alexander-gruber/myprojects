
import re
import string
import requests
import unicodedata
import numpy as np

def extract_doi(text):
    """
    Extract the first DOI from a text string.

    Handles formats such as:
        10.1234/abc.def
        doi:10.1234/abc.def
        DOI: 10.1234/abc.def
        https://doi.org/10.1234/abc.def
        http://doi.org/10.1234/abc.def
        https://dx.doi.org/10.1234/abc.def

    Returns:x
        str or None: DOI without the URL/prefix.
    """

    pattern = re.compile(
        r"""
        (?:
            https?://
        )?
        (?:
            (?:dx\.)?doi\.org/
        )?
        (10\.\d{4,9}/
            [-._;()/:A-Z0-9]+
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )
    match = pattern.search(text)
    if not match:
        return None
    doi = match.group(1)

    # Remove punctuation commonly appended to a DOI in references
    doi = doi.rstrip(".,;:!?)]}")

    return doi

def clean_text(text):
    """Basic normalization of text for BibTeX."""

    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def bibtex_escape(text):
    """Escape characters that have special meaning in BibTeX."""

    if not text:
        return ""
    text = clean_text(text)
    # Escape ampersands
    text = text.replace("&", r"\&")

    return text

def get_crossref_metadata(doi):
    """Retrieve metadata for a DOI from Crossref."""

    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": "citation-to-bibtex/1.0"}
    params = {}
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=10
    )
    response.raise_for_status()
    return response.json()["message"]

def make_citation_key(metadata):
    """Generate a BibTeX citation key as AuthorYYYY."""

    authors = metadata.get("author", [])
    if authors:
        surname = authors[0].get("family", "Unknown")
    else:
        surname = "Unknown"

    year = None
    for field in ["published-print", "published-online", "issued"]:
        if field in metadata:
            date_parts = metadata[field].get("date-parts", [])
            if date_parts:
                year = date_parts[0][0]
                break
    surname = re.sub(r"[^A-Za-z0-9]", "", surname)

    return f"{surname}{year or 'n.d.'}"

def create_bibtex_from_doi(doi, entries=None):

    if not entries:
        entries = []

    metadata = get_crossref_metadata(doi)
    fields = {}

    authors = metadata.get("author", [])
    if authors:
        author_strings = []
        for author in authors:
            family = author.get("family", "")
            given = author.get("given", "")
            if family and given:
                author_strings.append(f"{family}, {given}")
            elif family:
                author_strings.append(family)
        fields["author"] = " and ".join(author_strings)
    if metadata.get("title"):
        fields["title"] = metadata["title"][0]
    if metadata.get("container-title"):
        fields["journal"] = metadata["container-title"][0]
    if metadata.get("volume"):
        fields["volume"] = metadata["volume"]
    if metadata.get("issue"):
        fields["number"] = metadata["issue"]
    if metadata.get("page"):
        fields["pages"] = metadata["page"]
    year = None
    for field in ["published-print", "published-online", "issued"]:
        if field in metadata:
            date_parts = metadata[field].get("date-parts", [])
            if date_parts:
                year = date_parts[0][0]
                break
    if year:
        fields["year"] = str(year)
    fields["doi"] = metadata["DOI"]

    key = make_citation_key(metadata)
    key = ensure_unique_key(key, entries)

    lines = [f"@article{{{key},"]
    items = list(fields.items())
    for i, (field, value) in enumerate(items):
        value = bibtex_escape(value)
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"  {field} = {{{value}}}{comma}")
    lines.append("}")

    entries.append([key, "\n".join(lines)])

    return entries

def create_bibtex_from_text(text, entries=None):

    if not entries:
        entries = []

    yr_start = text.find('(')
    yr_end = text.find(')')

    authors = text[0:yr_start].strip()
    year = text[yr_start+1:yr_end]
    title = text[yr_end+3::]

    key = f"{authors.split(',')[0]}{year}"
    key = ensure_unique_key(key, entries)

    # Build BibTeX
    lines = [f"@report{{{key},"]
    lines.append(f"  author = {{{bibtex_escape(authors)}}},")
    lines.append(f"  title = {{{bibtex_escape(title)}}},")
    lines.append(f"  year = {{{bibtex_escape(year)}}}")
    lines.append("}")

    entries.append([key, "\n".join(lines)])

    return entries

def ensure_unique_key(key, entries):
    keys = np.array([e[0] for e in entries])
    ind = np.where(keys==key)[0]
    if len(ind) > 0:
        key += string.ascii_lowercase[len(ind)]
    return  key

def create_ismn_references(networks=None,
                          return_bibtex=True,
                          return_reftext=True):

    fname = r"C:\Users\agruber\Desktop\temp\all_citations_mainly_txt_july_07_2025.txt"
    bibfile= r"C:\Users\agruber\Desktop\temp\references.bib"
    txtfile= r"C:\Users\agruber\Desktop\temp\references.txt"

    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().split('\n\n')

    bibentries = []
    nets = []
    refs = []
    types = []
    keys = []

    # General ISMN references
    for ref in lines[0:2]:
        bibentries = create_bibtex_from_doi(extract_doi(ref), entries=bibentries)
    ismnkeys = [e[0] for e in bibentries]

    #  Network references
    for i, line in enumerate(lines[3::]):
        tmp = line.split('\n')
        # Correct erroneous line brake; happens for TWENTE (status August 2026)
        if len(tmp)>2:
            tmp = [tmp[0], " ".join(tmp[1:])]

        # Extract references for specific networks only
        net, ref = tmp[0][len('%network: ')::], tmp[1]
        if networks:
            if net not in networks:
                continue

        nets += [net]
        refs += [ref]

        # Distinguish references with and without bibtex entry.
        if 'We acknowledge the work of' in ref:
            types += ['Ack']
            keys += []
        else:
            types += ['Ref']
            if extract_doi(ref):
                bibentries = create_bibtex_from_doi(extract_doi(ref), entries=bibentries)
            else:
                bibentries = create_bibtex_from_text(ref, entries=bibentries)
            keys += [bibentries[-1][0]]

    nets = np.array(nets)
    refs = np.array(refs)
    types = np.array(types)
    keys = np.array(keys)

    if return_bibtex:
        with open(bibfile, "w", encoding="utf-8") as f:
            f.write("\n\n".join([e[1] for e in bibentries]))
            f.write("\n")

    if return_reftext:
        #  General ISMN acknowledgement
        reftext = f'We thank the ISMN for their continuous support of the community \\citep{{{','.join(ismnkeys)}}}. '

        # Acknowledgement text without bibtex entries
        ind_ack = np.where(types == 'Ack')[0]
        if len(ind_ack) > 0:
            reftext += 'We acknowledge the work '
            ack_start = len('We acknowledge the work ')
            ack_end = len('in support of the ISMN.')
            acks = refs[ind_ack]
            for i, ack in enumerate(acks):
                reftext += ack[ack_start:-(ack_end+1)]
                if i < len(acks)-2:
                    reftext += ', '
                elif i < len(acks)-1:
                    reftext += ', and '
            reftext += ' in support of the ISMN. '

        # Acknowledgement text with bibtex entries
        networks = np.unique(nets)
        refstrings = []
        cnt = 0
        for i, net in enumerate(networks):
            ind = np.where((nets == net) & (types == 'Ref'))
            if len(ind[0]) == 0:
                continue
            cnt += 1
            refstrings += [f'{net} \\citep{{{','.join(keys[ind])}}}']

        if cnt == 1:
            reftext += f'We thank the network {refstrings[0]} for sharing for sharing their data with the ISMN.'
        elif cnt > 1:
            reftext += 'We thank the networks '
            reftext += ', '.join(refstrings[0:len(refstrings)-1]) + ', and ' + refstrings[-1]
            reftext += ' for sharing their data with the ISMN.'
        else:
            pass

        with open(txtfile, "w", encoding="utf-8") as f:
            f.write(reftext)


if __name__=='__main__':

    create_ismn_references(networks=['AACES','ARM','AMMA-CATCH','RSMN','RU-ADDFerti'])

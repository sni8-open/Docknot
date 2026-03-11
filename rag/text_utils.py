import re


def clean_text(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def detect_section_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    patterns = [
        r"^\d+(\.\d+)*\s+.+",
        r"^[A-Z][A-Za-z0-9 ,\-\(\):]+$",
    ]

    for line in lines[:10]:
        for p in patterns:
            if re.match(p, line):
                return line

    return ""


def chunk_pages(page_entries: list[dict], chunk_size: int = 1800, overlap: int = 500):
    full_text = []
    page_boundaries = []
    cursor = 0

    for entry in page_entries:
        page_block = f"\n--- Page {entry['page_num']} ---\n{entry['text']}\n"
        full_text.append(page_block)
        start = cursor
        cursor += len(page_block)
        end = cursor
        page_boundaries.append((start, end, entry["page_num"]))

    text = "".join(full_text).strip()

    chunks = []
    i = 0
    n = len(text)
    chunk_index = 0

    while i < n:
        j = min(i + chunk_size, n)
        chunk_text = text[i:j].strip()

        if chunk_text:
            pages = []
            for start, end, page_num in page_boundaries:
                if not (j < start or i > end):
                    pages.append(page_num)

            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None
            section_title = detect_section_title(chunk_text)

            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": section_title,
                "char_length": len(chunk_text),
                "preview": chunk_text[:180]
            })
            chunk_index += 1

        if j == n:
            break

        i = max(0, j - overlap)

    return chunks
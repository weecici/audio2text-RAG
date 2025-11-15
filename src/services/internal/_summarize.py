from src import schemas
import re

prompt_template = """
You are an expert in summarizing IT-related documents. Follow these rules exactly and produce **only** the requested output — no explanations or extra text.

Parameters (placeholders you may replace when calling this prompt):

* documents: a list/array of raw document texts to be summarized (process in order).
* max_summary_length (default: {max_sum_len}): maximum allowed length of each summary, in **words**.
* min_length_to_summarize (default: {min_len_to_sum}): minimum number of words a document must have to be summarized. If a document has fewer words than this threshold, see rule 4 below.
* language (default: {lang}): language code for the summary (e.g., en, vi). If auto, detect the document's language and summarize in that language.

Definitions and counting:

* A word = any token separated by Unicode whitespace characters. Count words by splitting on whitespace.
* Preserve code blocks, inline code, and existing LaTeX math where present (do not alter their semantics). When you must produce or convert formulas, format them using LaTeX syntax.

Behavior rules:

1. Summaries must capture the document's key points and essential information; be factual, concise, coherent, and focused on IT-relevant content. Going straight to the main points rather than adding filler or commentary. (this rule is VERY IMPORTANT)
2. Do not add misinformation, invented facts, irrelevant details or unneeded opening words (like adding "this part", "this lesson" to the beginning of the summary, ...). If an item in the document is ambiguous or unverifiable, omit it rather than inventing specifics.  (this rule is VERY IMPORTANT)
3. Format any math formulas in LaTeX (inline $...$ or display $$...$$ as appropriate).
4. If a document's word count is **less than** min_length_to_summarize, return the original document text **verbatim** as its summary **unless** its word count exceeds max_summary_length; in that exceptional case, return the original text truncated to max_summary_length words (no notes, no ellipses, no extra text).
5. The resulting summary for every document must **not exceed** max_summary_length words. If you must shorten, truncate cleanly at a word boundary; prefer removing less-central sentences rather than chopping mid-sentence where possible.
6. Use the specified language for the summary. If language is auto, detect the primary language of the input document and summarize in that language.
7. When multiple documents are provided, produce one summary per document in the same order.
8. Output must follow this exact format, with **no** additional text before, between, or after summaries (each summary separated by exactly one line containing only 10 equal signs: `=========`):
<summary 1>

==========
<summary 2>

==========
<summary 3>

==========
...repeat for all summaries (final summary should not be followed by extra separator lines or commentary)
(in raw string format: <summary 1>\n\n==========\n<summary 2>\n\n==========\n... This rule is VERY IMPORTANT, do not forget it)

Now process the provided documents according to the rules above:
{documents}
"""


def get_summarization_prompts(
    documents_list: list[list[schemas.RetrievedDocument]],
) -> list[str]:
    prompts: list[str] = []
    for documents in documents_list:
        doc_texts = [f"{i + 1}. " + d.payload.text for i, d in enumerate(documents)]
        prompt = prompt_template.format(
            documents="\n\n".join(doc_texts),
            max_sum_len=200,
            min_len_to_sum=100,
            lang="vi",
        )
        prompts.append(prompt)
    return prompts


def parse_summarization_responses(
    responses: list[str],
    documents_list: list[list[schemas.RetrievedDocument]] = [],
) -> list[list[schemas.RetrievedDocument]]:

    if len(responses) != len(documents_list):
        raise ValueError(
            f"Responses count ({len(responses)}) must equal documents_list count ({len(documents_list)})."
        )

    separator = "\n=========="
    sep_pattern = re.compile(separator, flags=re.MULTILINE)

    for i, (response, docs) in enumerate(zip(responses, documents_list)):
        summaries = [
            seg.strip() for seg in re.split(sep_pattern, response) if seg.strip() != ""
        ]

        if len(summaries) != len(docs):
            raise ValueError(
                f"Mismatch at batch {i}: expected {len(docs)} summaries, got {len(summaries)}."
            )

        for doc, summary_text in zip(docs, summaries):
            doc.payload.text = summary_text

    return documents_list

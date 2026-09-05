from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# CONFIGURATION
# ==================================================

FAQ_PATH = Path("data/college_faq.txt")

RELEVANCE_THRESHOLD = 0.20


# ==================================================
# LOAD FAQ
# ==================================================

def load_faq():
    """Load the FAQ knowledge base."""

    if not FAQ_PATH.exists():
        raise FileNotFoundError(
            f"FAQ file not found: {FAQ_PATH}"
        )

    return FAQ_PATH.read_text(
        encoding="utf-8"
    )


# ==================================================
# SPLIT FAQ INTO CHUNKS
# ==================================================

def split_into_chunks(text):
    """Split the FAQ document into Q&A blocks."""

    chunks = []

    for block in text.split("\n\n"):

        block = block.strip()

        if block and block.startswith("Q:"):
            chunks.append(block)

    return chunks


# ==================================================
# CLEAN QUESTION
# ==================================================

def clean_question(question):
    """
    Remove common conversational filler from
    natural speech.
    """

    filler_phrases = [
        "hello",
        "hi",
        "hey",
        "this is a test",
        "this is the test",
        "can you tell me",
        "could you tell me",
        "please tell me",
        "i want to know",
        "i would like to know",
    ]

    cleaned = question.lower()

    for phrase in filler_phrases:

        cleaned = cleaned.replace(
            phrase,
            " "
        )

    return " ".join(
        cleaned.split()
    )


# ==================================================
# QUERY EXPANSION
# ==================================================

def expand_query(question):
    """
    Add FAQ terminology to natural-language questions.

    We ADD related terminology instead of replacing
    the original question.
    """

    expanded = question.lower()

    # ----------------------------------------------
    # COLLEGE WORKING HOURS
    # ----------------------------------------------

    opening_patterns = [
        "opens",
        "open",
        "opening",
        "starts",
        "begin",
        "begins",
    ]

    closing_patterns = [
        "closes",
        "close",
        "closing",
        "ends",
        "end",
    ]

    has_opening = any(
        word in expanded
        for word in opening_patterns
    )

    has_closing = any(
        word in expanded
        for word in closing_patterns
    )

    if has_opening or has_closing:

        expanded += (
            " college working hours "
            "college timings opening time "
            "closing time"
        )


    # ----------------------------------------------
    # EXAMINATION TIMETABLE
    # ----------------------------------------------

    if any(
        phrase in expanded
        for phrase in [
            "exam schedule",
            "exam timetable",
            "exam time",
            "when are exams",
            "examination schedule",
        ]
    ):

        expanded += (
            " examination timetable "
            "examination department"
        )


    # ----------------------------------------------
    # EXAMINATION FEES
    # ----------------------------------------------

    if any(
        word in expanded
        for word in [
            "exam fee",
            "exam fees",
            "examination fee",
            "examination fees",
        ]
    ):

        expanded += (
            " examination fees "
            "examination department"
        )


    # ----------------------------------------------
    # BONAFIDE CERTIFICATE
    # ----------------------------------------------

    if any(
        phrase in expanded
        for phrase in [
            "bonafide",
            "bonafide document",
            "proof of student",
        ]
    ):

        expanded += (
            " bonafide certificate "
            "administrative office"
        )


    # ----------------------------------------------
    # LEAVING CERTIFICATE
    # ----------------------------------------------

    if any(
        phrase in expanded
        for phrase in [
            "leaving certificate",
            "college leaving",
            "lc",
        ]
    ):

        expanded += (
            " leaving certificate "
            "administrative office"
        )


    # ----------------------------------------------
    # LIBRARY
    # ----------------------------------------------

    if "library" in expanded:

        expanded += (
            " library working hours "
            "library books borrowing"
        )


    # ----------------------------------------------
    # ADMISSION
    # ----------------------------------------------

    if any(
        phrase in expanded
        for phrase in [
            "admission",
            "apply",
            "application",
            "join college",
        ]
    ):

        expanded += (
            " admission application "
            "admission office"
        )


    # ----------------------------------------------
    # ATTENDANCE
    # ----------------------------------------------

    if any(
        word in expanded
        for word in [
            "attendance",
            "absent",
            "presence",
        ]
    ):

        expanded += (
            " attendance requirement "
            "minimum attendance"
        )


    return " ".join(
        expanded.split()
    )


# ==================================================
# QUESTION INTENT
# ==================================================

def extract_question_intent(question):
    """
    Extract important intent words from a question.
    """

    intent_keywords = {
        "refund",
        "return",
        "cancel",
        "cancellation",

        "pay",
        "payment",
        "fee",
        "fees",

        "timetable",
        "schedule",

        "hours",
        "timing",
        "opening",
        "closing",

        "location",
        "where",
        "contact",

        "documents",
        "admission",
        "apply",
        "application",

        "attendance",

        "certificate",
        "bonafide",
        "leaving",

        "borrow",
        "library",

        "academic",
        "records",
        "problem",
    }

    words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            question.lower()
        )
    )

    return words.intersection(
        intent_keywords
    )


# ==================================================
# RETRIEVE FAQ
# ==================================================

def retrieve(question, top_k=3):
    """
    Retrieve relevant FAQ entries using:

    - Speech cleanup
    - Query expansion
    - Word TF-IDF
    - Character TF-IDF
    - Intent matching
    """

    text = load_faq()

    chunks = split_into_chunks(text)

    if not chunks:
        return []


    # ----------------------------------------------
    # CLEAN QUESTION
    # ----------------------------------------------

    cleaned_question = clean_question(
        question
    )


    # ----------------------------------------------
    # EXPAND QUESTION
    # ----------------------------------------------

    expanded_question = expand_query(
        cleaned_question
    )


    # ----------------------------------------------
    # WORD TF-IDF
    # ----------------------------------------------

    documents = chunks + [
        expanded_question
    ]

    word_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    word_matrix = word_vectorizer.fit_transform(
        documents
    )

    word_question = word_matrix[-1]

    word_chunks = word_matrix[:-1]

    word_scores = cosine_similarity(
        word_question,
        word_chunks
    ).flatten()


    # ----------------------------------------------
    # CHARACTER TF-IDF
    # ----------------------------------------------

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    char_matrix = char_vectorizer.fit_transform(
        documents
    )

    char_question = char_matrix[-1]

    char_chunks = char_matrix[:-1]

    char_scores = cosine_similarity(
        char_question,
        char_chunks
    ).flatten()


    # ----------------------------------------------
    # COMBINE SCORES
    # ----------------------------------------------

    combined_scores = (
        0.7 * word_scores
        + 0.3 * char_scores
    )


    # ----------------------------------------------
    # INTENT MATCHING
    # ----------------------------------------------

    question_intent = extract_question_intent(
        expanded_question
    )

    adjusted_scores = combined_scores.copy()


    for index, chunk in enumerate(chunks):

        chunk_intent = extract_question_intent(
            chunk
        )

        missing_intent = (
            question_intent - chunk_intent
        )


        # Strong intents that should not be
        # ignored during retrieval.
        strong_missing_intent = {
            "refund",
            "return",
            "cancel",
            "cancellation",
            "borrow",
            "documents",
            "attendance",
            "bonafide",
            "leaving",
        }


        if missing_intent.intersection(
            strong_missing_intent
        ):

            adjusted_scores[index] *= 0.35


    # ----------------------------------------------
    # RANK RESULTS
    # ----------------------------------------------

    ranked_indices = (
        adjusted_scores.argsort()[::-1]
    )

    results = []


    for index in ranked_indices[:top_k]:

        if (
            adjusted_scores[index]
            >= RELEVANCE_THRESHOLD
        ):

            results.append(
                {
                    "text": chunks[index],
                    "score": float(
                        adjusted_scores[index]
                    )
                }
            )


    return results


# ==================================================
# DIRECT RAG TEST
# ==================================================

if __name__ == "__main__":

    question = input(
        "Ask a question: "
    ).strip()


    cleaned = clean_question(
        question
    )

    expanded = expand_query(
        cleaned
    )


    print(
        "\nOriginal question:"
    )

    print(question)


    print(
        "\nExpanded question:"
    )

    print(expanded)


    results = retrieve(question)


    print(
        "\n--- RETRIEVED INFORMATION ---"
    )


    if not results:

        print(
            "No relevant information found."
        )

    else:

        for result in results:

            print(
                f"\nScore: "
                f"{result['score']:.3f}"
            )

            print(
                result["text"]
            )
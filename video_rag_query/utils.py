import re
from typing import Set

# Simple list of English stop words to filter out common functional words
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", 
    "from", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
    "don", "should", "now", "is", "was", "were", "be", "been", "being", "have", 
    "has", "had", "having", "do", "does", "did", "doing", "what", "which", "who", "whom"
}

def extract_keywords(text: str) -> Set[str]:
    """
    Extract unique, meaningful keywords from a text string.
    Lowercases, removes punctuation, and filters stop words.
    """
    if not text:
        return set()
    
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Split into words and filter
    words = text.split()
    keywords = {w for w in words if w not in STOP_WORDS and len(w) > 2}
    
    return keywords

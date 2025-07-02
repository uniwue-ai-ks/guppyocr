import wikipedia
import random

def get_random_page_content() -> str:
    text = random.choice(wikipedia.page(title=random.choice(words)).section
    try:
        page_content = wikipedia.page(page_title).summary
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        return get_random_page_content()
    return page_content





for i in range(10):
    print(get_random_page_content())
    print("\n\n")

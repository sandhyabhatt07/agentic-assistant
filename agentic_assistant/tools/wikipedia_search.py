"""Wikipedia summary lookup."""

from __future__ import annotations

import logging
import warnings

import wikipedia

from .base import Tool, ToolError

logger = logging.getLogger(__name__)

USER_AGENT = "agentic-assistant/1.0 (https://github.com/sandhyabhatt07/agentic-assistant)"

# The wikipedia package triggers a BeautifulSoup "no parser specified" warning on disambiguation pages
warnings.filterwarnings("ignore", message=".*No parser was explicitly specified.*")


def make_wikipedia_tool(sentences: int = 2, language: str = "en") -> Tool:
    wikipedia.set_lang(language)
    # The library defaults to plain-HTTP API URLs; use HTTPS and identify the client per Wikimedia's API policy
    wikipedia.wikipedia.API_URL = f"https://{language.lower()}.wikipedia.org/w/api.php"
    wikipedia.set_user_agent(USER_AGENT)

    def search(query: str) -> str:
        query = query.strip()
        if not query:
            raise ToolError("empty Wikipedia query")
        try:
            # auto_suggest often rewrites good queries into the wrong page, so search for a title first
            titles = wikipedia.search(query, results=1)
            if not titles:
                raise ToolError(f"no Wikipedia page found for '{query}'")
            return str(wikipedia.summary(titles[0], sentences=sentences, auto_suggest=False))
        except wikipedia.exceptions.DisambiguationError as exc:
            options = ", ".join(str(o) for o in exc.options[:5])
            raise ToolError(f"multiple results found, please be more specific. Options: {options}") from None
        except wikipedia.exceptions.PageError:
            raise ToolError(f"no Wikipedia page found for '{query}'") from None
        except ToolError:
            raise
        except Exception as exc:  # network failures, JSON errors from the API
            logger.debug("Wikipedia lookup for %r failed", query, exc_info=True)
            raise ToolError(f"could not reach Wikipedia right now ({type(exc).__name__}), try again later") from None

    return Tool(
        name="wikipedia",
        description=(
            "Searches Wikipedia and returns a short summary. Use for factual information, definitions, "
            "historical facts and biographies. Input should be the topic only, e.g. 'Albert Einstein'."
        ),
        func=search,
    )

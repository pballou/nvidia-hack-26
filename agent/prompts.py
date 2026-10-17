"""Single source for the Lightning system prompt and the tool schema.

Used by the interactive planner (agent/planner.py) and the Gym task rows
(resources_servers/web_agent/make_tasks.py), so both see exactly the same tools.

Target descriptions are what the grounder VLM is trained on (T3's rewrite styles):
short, specific, visual and positional, e.g. "the search icon in the top-right toolbar"
or "the Edit button in the second row of the Orders table".

Every example here is deliberately unlike the sandbox tasks (agent-v4): an example that
names an eval target (agent-v3 had "the star next to the message from the landlord")
leaks the answer and inflates absolute agent numbers.
"""

from __future__ import annotations

import copy
import hashlib
import json

PROMPT_VERSION = "agent-v4"

TOOL_NAMES = ("click", "type_text", "press_key", "scroll", "done")

_TARGET_HELP = (
    "A short visual description of ONE element on the current screen, as a sighted person "
    "would point it out: its visible text or icon, its type, and where it is. Examples: "
    "\"the search icon in the top-right toolbar\", \"the Edit button in the second row of the "
    "Orders table\", \"the Wi-Fi toggle\", \"the Weekly option in the Repeat control\", \"the "
    "Paris pin on the map\". Use the words from the "
    "screen description, written as a phrase like these examples (not in its \"type: text @ location\" "
    "line format); never use coordinates, CSS selectors or element ids."
)

# Chat-completions tool format ({"type": "function", "function": {...}}).
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click one element on the screen. " + _TARGET_HELP,
            "parameters": {
                "type": "object",
                "properties": {"target_description": {"type": "string", "description": _TARGET_HELP}},
                "required": ["target_description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text with the keyboard. If target_description is given, that field is clicked "
                "first and its current text is replaced; otherwise the text goes to the focused field. "
                "Does not press Enter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The exact text to type."},
                    "target_description": {
                        "type": "string",
                        "description": "Optional: the input field to type into, e.g. \"the Subject field\".",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press one key or key combination, e.g. Enter, Tab, Escape, Backspace, ArrowDown, PageDown.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "Key name, e.g. \"Enter\" or \"Escape\"."}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page to reveal content that is off screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount": {"type": "string", "enum": ["small", "page"],
                               "description": "small = a few lines, page = about one screen."},
                },
                "required": ["direction", "amount"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Finish the task. Call it once the screen shows the request is complete (success=true), "
                "or when it cannot be done (success=false). Call no tools after this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One or two sentences for the user: what you did."},
                    "success": {"type": "boolean", "description": "Whether the request is now complete."},
                },
                "required": ["summary", "success"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are GUI Assist, an accessibility agent. A user who has trouble using apps tells you \
what they want, and you do it for them in the app on their screen.

You cannot see the screen. After every action you get a text observation: the page URL \
and title, whether the screen changed, the result of your last action, and a description \
of the visible screen written by a vision model. You act only through the tools; a vision \
model finds the element you describe and clicks it.

How to work:
- Until the task is done, every reply is exactly one tool call, never plain text. Then read \
the new observation.
- Describe click targets the way you would point them out to a sighted helper: visible \
text or icon, element type, and position, e.g. "the search icon in the top-right toolbar". \
Use the words from the screen description, but as a natural phrase: the line \
"- button: Save @ bottom-right" becomes "the Save button at the bottom right". Never copy \
the "@" notation.
- Use exactly the argument names of each tool, e.g. click(target_description=...).
- Icons often have no text; describe their shape and location ("the magnifier icon", "the \
trash can icon in the second row").
- If the screen did not change after a click, the click probably missed: describe the \
target differently (more specific, add its position) or try another way.
- If what you need is not visible, open the menu, tab or page where it likely is, or scroll.
- Do only what the user asked. Don't change other settings or submit things they didn't ask for.
- STATE in the description shows current values. To change a setting, pick the option the \
user wants, not the one already selected: "bigger" or "more" means an option past the \
current one in that direction.
- Before done(success=true), check that the latest observation (STATE, a toast or the \
page) shows exactly the result the user asked for; if it shows something else, keep working.
- When the observation shows the request is done, call done(summary, success=true); don't \
just say so in text. If it is impossible, call done with success=false and say why. After \
done, call no more tools."""


def initial_user_message(request: str, observation_text: str) -> str:
    """First user turn: the user's request plus the starting observation."""
    return f"User request: {request.strip()}\n\nCurrent screen:\n{observation_text}"


def prompt_sha(text: str) -> str:
    """Short stable id of a prompt (first 10 hex chars of sha256), recorded in trace events."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


# The exact system prompt + tool schema Lightning sees; changes whenever either does.
PROMPT_SHA = prompt_sha(SYSTEM_PROMPT + json.dumps(TOOLS, sort_keys=True))


def chat_tools() -> list[dict]:
    """Tools in chat-completions format (a deep copy, safe to mutate)."""
    return copy.deepcopy(TOOLS)


def responses_tools(strict: bool = False) -> list[dict]:
    """Tools in Responses-API format, for Gym rows: {type, name, description, parameters, strict}."""
    out = []
    for tool in TOOLS:
        fn = tool["function"]
        out.append({
            "type": "function",
            "name": fn["name"],
            "description": fn["description"],
            "parameters": copy.deepcopy(fn["parameters"]),
            "strict": strict,
        })
    return out


def tool_parameters(name: str) -> dict:
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool["function"]["parameters"]
    raise KeyError(name)

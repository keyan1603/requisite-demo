"""
09 - Prompt templates: reusable, parameterized prompts, dependency-free
(plain str.format under the hood).

Also verifies two fixes from ADR-0031: dotted-field validation, and
partial()'s brace-escaping (a value can no longer inject a new
placeholder for a later .format() call to unexpectedly fill).

Run with:
    python demos/09_prompts.py
"""

from _shared import make_ai
from requisite.prompts import ChatPromptTemplate, PromptTemplate


def basic_template() -> None:
    print("=== PromptTemplate ===")
    template = PromptTemplate.from_template("Translate to {language}: {text}")
    print("input_variables:", template.input_variables)
    print(template.format(language="French", text="Hello, how are you?"))

    french_translator = template.partial(language="French")
    print(french_translator.format(text="Good morning"))


def chat_template_with_ai() -> None:
    print("\n=== ChatPromptTemplate -> list[Message] -> ai.chat(...) ===")
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {persona}. Answer in one short sentence."),
            ("user", "{question}"),
        ]
    )
    messages = chat_template.format_messages(
        persona="pirate", question="Where do AI agents keep their treasure?"
    )
    ai = make_ai()
    print(ai.chat(messages))


def adr_0031_regression_checks() -> None:
    print("\n=== ADR-0031 regression checks ===")

    # 1. A dotted field's root is now correctly required as an input
    #    variable, and a missing root raises a clean PromptException
    #    (previously a raw KeyError).
    from requisite.core.exceptions import PromptException

    class Cfg:
        api_key = "sk-demo"

    dotted = PromptTemplate.from_template("Config: {cfg.api_key}")
    assert dotted.input_variables == ["cfg"]
    try:
        dotted.format()
        raise AssertionError("expected PromptException")
    except PromptException as exc:
        print(f"  missing dotted-field root raises cleanly: {exc}")
    print(f"  resolved: {dotted.format(cfg=Cfg())}")

    # 2. partial() no longer lets a substituted string value inject a
    #    new placeholder for a later .format() call to fill.
    injection_template = PromptTemplate.from_template("Hello {name}, secret is {secret}.")
    partially_filled = injection_template.partial(name="{secret}")
    result = partially_filled.format(secret="TOP-SECRET-VALUE")
    assert result == "Hello {secret}, secret is TOP-SECRET-VALUE.", result
    print(f"  partial() injection closed: {result!r}")


def main() -> None:
    basic_template()
    chat_template_with_ai()
    adr_0031_regression_checks()


if __name__ == "__main__":
    main()

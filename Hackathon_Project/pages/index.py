"""The overview page of the app."""

import reflex as rx
from .. import styles
from ..templates import template
from ..views.stats_cards import stats_cards


from ..components.card import card
import datetime


def _time_data() -> rx.Component:
    return rx.vstack(
        rx.text("""About The Project""", size="8", weight="medium", underlined="always"),
        rx.text("""SpeechReclaim is a software that takes in user input for a set of diseases and computes whether the user is suffering from a neurological speech disorder or not. The neurological disorders that this software currently focuses on are Parkinson's speech disorder, Stuttering, and Aphasia. Based on the prediction about the disease, a vital feature that we aim at implementing is an AI chatbot which gives a long term training plan suggestion to the user which can potentially help them combat or overcome their disorder.""", size="5", weight="medium",align="center"),
        align="center",
        spacing="2",
    )


def tab_content_header() -> rx.Component:
    return rx.hstack(
        _time_data(),
        align="center",
        width="100%",
        spacing="4",
    )


@template(route="/", title="About")
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.
    """
    return rx.vstack(
        rx.flex(
            justify="between",
            align="center",
            width="100%",
        ),
        card(
            rx.hstack(
                tab_content_header(),
                width="100%",
                justify="between",
            ),
        ),
        spacing="8",
        width="100%",
    )

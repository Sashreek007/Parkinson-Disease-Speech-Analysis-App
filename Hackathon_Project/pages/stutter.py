"""The overview page of the app."""

import reflex as rx
from .. import styles
from ..templates import template
from ..views.stats_cards import stats_cards
import datetime

from ..components.card import card
from ..backend.Stutter_State import StutterState

def tab_content_header() -> rx.Component:
    return rx.hstack(
        rx.text("""Stuttering
Treatment: 
Currently there is no specific medication that has been proven to help this condition however alternative solutions such as treatments for stuttering include speech therapy, cognitive behavioral therapy, and electronic devices. All these treatments focus on allowing an individual to show their speech or decrease possible anxiety and stress which can make stuttering worse. Some exploratory methods that are being tested are the use of virtual reality and computer based programs to simulate social environments. This will allow the individual to practice in a controlled environment and reduce anxiety.
""", size="2", weight="medium"),
        align="center",
        spacing="2",
        display=["none", "none", "flex"],
    )

@template(route="/stutter", title="Stutter")
def stutter() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.
    """
    
    return rx.vstack(
        rx.flex(
            rx.heading("Stutter Health Status Prediction"),
            rx.upload(
                rx.vstack(
                    rx.button(
                        "Select Audio File",
                        bg="white",
                        border="1px solid rgb(107,99,246)",
                    ),
                    rx.text(
                        "Drag and drop audio files here or click to select files"
                    ),
                ),
                id="audio_upload",
                accept={
                    "audio/*": [".wav", ".mp3"]
                },
                multiple=False,
                border="1px dotted rgb(107,99,246)",
                padding="5em",
            ),
            rx.button(
                "Process Audio",
                on_click=StutterState.handle_upload(
                    rx.upload_files(upload_id="audio_upload")
                ),
            ),
            rx.heading(StutterState.prediction),
            justify="between",
            align="center",
            width="100%",
            padding="5em",
        ),
        card(
            rx.hstack(
                tab_content_header(),
            ),
        ),
        spacing="8",
        width="100%",
    )

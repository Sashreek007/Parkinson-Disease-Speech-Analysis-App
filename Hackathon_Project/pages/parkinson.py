"""The overview page of the app."""

import reflex as rx
from .. import styles
from ..templates import template
from ..views.stats_cards import stats_cards
import datetime

from ..components.card import card
from ..backend.Park_State import ParkState



@template(route="/parkinson", title="Parkinson's Speech Disorder")
def parkinson() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.
    """
    
    return rx.vstack(
        rx.flex(
            rx.vstack(  # Wrap content in vstack for vertical alignment
                rx.spacer(),
                rx.spacer(),
                rx.spacer(),
                rx.heading("Parkinson's Health Status Prediction"),
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
                    align="center",  # Centers items horizontally
                    justify="center", # Centers items vertically
                    width="100%",
                ),
                id="audio_upload",
                accept={
                    "audio/*": [".wav", ".mp3"]
                },
                multiple=False,
                border="1px dotted rgb(107,99,246)", 
                padding="10em",
            ),
                rx.button(
                    "Process Audio",
                    on_click=ParkState.handle_upload(
                        rx.upload_files(upload_id="audio_upload")
                    ),
                ),
                rx.heading(ParkState.prediction),
                align="center",  # Centers items horizontally
                spacing="2",    # Adds space between items
            ),
            width="100%",
            justify="center",  # Centers the vstack in the flex container
            align="center",    # Centers items vertically
    ),
        spacing="8",
        width="100%",
        justify="between",
        align="center",
    )

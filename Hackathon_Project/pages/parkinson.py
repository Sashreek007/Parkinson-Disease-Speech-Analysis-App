"""The overview page of the app."""

import reflex as rx
from .. import styles
from ..templates import template
from ..views.stats_cards import stats_cards
import datetime

from ..components.card import card
from ..backend.Park_State import ParkState

def tab_content_header() -> rx.Component:
    return rx.hstack(
        rx.text("""Parkinson’s
Treatment: 
For mild cases of parkinson’s treatment can include certain medications such as Levodopa, which is considered the most effective parkinson’s disease medicine, in combination with Caribidopa which can reduce the side effects that often come with Levodopa. Other medications are split into three categories: dopamine agonists, Monoamine oxidase B inhibitors, and Catechol O-methyltransferase inhibitors. With some possible medication being rotigotine, pramipexole, rasagiline, selegiline and more. Alternatively, other solutions incorporate regular exercise, a healthy and balanced diet, and physical and occupational therapy.

For more severe cases surgical treatments such as deep brain stimulation(DBS) can be considered. DBS involves specific placement of electrodes within the brain to control and reduce motor issues like tremors, and dyskinesia, that is involuntary muscle movements.
""", size="4", weight="medium",align="center"),
        align="center",
        justify="between",
        spacing="2",
        display=["none", "none", "flex"],
    )

@template(route="/parkinson", title="Parkinson's Speech Disorder")
def parkinson() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.
    """
    
    return rx.vstack(
        rx.flex(
            rx.vstack(  # Wrap content in vstack for vertical alignment
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
                padding="5em",
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
        card(
            rx.hstack(
                tab_content_header(),
            ),
        ),
        spacing="8",
        width="100%",
        justify="between",
        align="center",
    )

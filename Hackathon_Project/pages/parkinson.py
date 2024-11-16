"""The overview page of the app."""

import reflex as rx
from .. import styles
from ..templates import template
from ..views.stats_cards import stats_cards
from ..views.charts import (
    users_chart,
    revenue_chart,
    orders_chart,
    area_toggle,
    StatsState,
)

from ..components.card import card
from .profile import ProfileState
from ..backend.Park_State import ParkState

def tab_content_header() -> rx.Component:
    return rx.hstack(
        area_toggle(),
        align="center",
        width="100%",
        spacing="4",
    )


@template(route="/parkinson", title="Parkinson", on_load=StatsState.randomize_data)
def parkinson() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.
    """
    
    return rx.vstack(
        rx.heading(f"Welcome, {ProfileState.profile.name}", size="5"),
        rx.flex(
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
            rx.flex(
                rx.button("Audio Input"),
                spacing="4",
                width="100%",
                wrap="nowrap",
                justify="end",
            ),
            justify="between",
            align="center",
            width="100%",
            padding="5em",
        ),
        stats_cards(),
        card(
            rx.hstack(
                tab_content_header(),
                rx.segmented_control.root(
                    rx.segmented_control.item("Users", value="users"),
                    rx.segmented_control.item("Revenue", value="revenue"),
                    rx.segmented_control.item("Orders", value="orders"),
                    margin_bottom="1.5em",
                    default_value="users",
                    on_change=StatsState.set_selected_tab,
                ),
                width="100%",
                justify="between",
            ),
            rx.match(
                StatsState.selected_tab,
                ("users", users_chart()),
                ("revenue", revenue_chart()),
                ("orders", orders_chart()),
            ),
        ),
        spacing="8",
        width="100%",
    )

"""The overview page of the app."""

import reflex as rx
from .. import styles
from ..templates import template
from ..views.stats_cards import stats_cards


from ..components.card import card
import datetime


def _time_data() -> rx.Component:
    return rx.vstack(
        rx.text("""About The Project""", size="8", weight="medium", underlined="always", align="center"),
        rx.text("""SpeechReclaim is a software that takes in user input for a set of diseases and computes whether the user is suffering from a neurological speech disorder or not. The neurological disorders that this software currently focuses on are Parkinson's speech disorder, Stuttering, and Aphasia. Based on the prediction about the disease, a vital feature that we aim at implementing is an AI chatbot which gives a long term training plan suggestion to the user which can potentially help them combat or overcome their disorder.""", size="5", weight="medium",align="center"),
        rx.spacer(),
        rx.text("""The current verison of the software can successfully predict if the user is suffering from Parkinson's Speech Disorder""", size="5", weight="medium", align="center"),
        rx.text("Parkinson's Speech Disorder", size="8", weight="medium",align="left"),
        rx.text("""Parkinson's disease, a progressive neurological disorder, often affects a person's speech. This condition, known as hypokinetic dysarthria, is caused by the degeneration of dopamine-producing neurons in the brain, leading to impaired motor control.""", size="5", weight="medium",align="center"),
        rx.text("""Key symptoms include:""", size="8", weight="medium",align="left"),
        rx.text("""Soft or Muffled Speech: The voice may sound quieter, breathy, or monotone.
Slurred Words: Reduced coordination of the vocal muscles can make articulation unclear.
Speech Pauses: Difficulty initiating speech or long pauses between words.
Fast or Hesitant Pace: Speech may become rushed or interrupted by stammering.
These changes can significantly impact communication, reducing a person’s confidence and social interactions. Early intervention with speech therapy, vocal exercises, and sometimes assistive technologies can help mitigate these effects and improve clarity and expression.""", size="5", weight="medium",align="center"),
        rx.text("""Additionally we also aim to train this model to identify by many more speech disorders like stuttering, aphrasia etc.""", size="5", weight="medium",align="center"),
        rx.text("Stuttering", size="8", weight="medium",align="left"),
        rx.text("""Stuttering, is a speech disorder characterized by interruptions in the normal flow of speech. These interruptions, or disfluencies, can include repeating sounds, syllables, or words, prolonging sounds, and experiencing blocks where no sound comes out despite an effort to speak. Stuttering often occurs alongside physical behaviors like blinking, jaw tension, or hand movements.
Researchers currently believe that stuttering is caused by a combination of factors, including genetics, language development, environment, as well as brain structure and function. Working together, these factors can influence the speech of a person who stutters.""", size="5", weight="medium", align="center"),
        rx.text("Aphasia", size="8", weight="medium",align="left"),
        rx.text("""Aphasia is a language disorder caused by damage to specific areas of the brain, typically in the left hemisphere, which controls language. It affects a person’s ability to speak, understand spoken language, read, or write but does not impact intelligence. The severity and type of aphasia depend on the location and extent of the brain injury.""", size="5", weight="medium", align="center"),

        rx.text("""The most common cause of aphasia is a stroke, but it can also result from traumatic brain injuries, tumors, or neurodegenerative conditions. Aphasia can be categorized into different types, including:""", size="5", weight="medium", align="center"),

        rx.text("""Broca's Aphasia: Difficulty in producing speech, often with effortful, halting language, but comprehension remains intact.""", size="5", weight="medium", align="center"),
        rx.text("""Wernicke's Aphasia: Fluent but nonsensical speech and significant difficulty understanding language.""", size="5", weight="medium", align="center"),
        rx.text("""Global Aphasia: Severe impairment in both speech production and comprehension, usually due to extensive brain damage.""", size="5", weight="medium", align="center"),
        rx.text("""Treatment involves speech and language therapy, focusing on relearning communication skills and developing alternative methods of expression. The recovery process can be gradual and varies widely among individuals, depending on the cause and extent of the brain damage.""", size="5", weight="medium", align="center"),
        rx.text("""Aphasia significantly impacts quality of life, but with proper therapy, support, and patience, individuals can often improve their communication abilities and regain independence.""", size="5", weight="medium",align="center"),
        spacing="6",
        align="center", 
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

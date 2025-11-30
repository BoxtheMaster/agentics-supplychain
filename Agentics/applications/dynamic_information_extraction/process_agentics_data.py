import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import asyncio
import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agentics import AG


class Answer(BaseModel):
    short_answer: str | None = None
    answer_report: str | None = Field(
        None,
        description="""
A detailed Markdown Document reporting evidence for the above answer
""",
    )
    evidence: list[str] | None = Field(
        None,
        description="""
list of evidence sources used to support the answer
""",
    )
    confidence: float | None = None


def data_processing():

    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    # if "dataset_index" not in st.session_state:
    #     st.session_state.dataset_index = {
    #         s.Date: i for i, s in enumerate(st.session_state.dataset.states)
    #     }

    uploaded_file = st.file_uploader("Upload your data", type=["csv", "json"])
    if uploaded_file is not None:
        st.success(f"File `{uploaded_file.name}` uploaded successfully!")
        if uploaded_file.name.endswith(".csv"):
            st.session_state.dataset = AG.from_csv(uploaded_file)
        if uploaded_file.name.endswith(".json"):
            st.session_state.dataset = AG.from_jsonl(uploaded_file)

    start = 0
    end = len(st.session_state.dataset) if st.session_state.dataset else 1

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        Select data to be analyzed
        """
        )
        st.markdown(f"Selected Model: **{st.session_state.selected_model}**")
        with st.form("Selection"):

            question = st.text_area("Define the subject of your analysis")

            selected_start = st.number_input(
                "First State Analyzed", min_value=start, max_value=end, value=1
            )
            selected_end = st.number_input(
                "Last State Analyzed", min_value=start, max_value=end - 1, value=end - 1
            )
            # type_code = st.text_area("Insert a Pydantic Type to focus your analysis", value=example_class, height=300)
            batch_size = st.number_input(
                "Analyze data in batches of",
                value=10,
                min_value=2,
                max_value=50,
                step=1,
            )

            market_sentiment_analysis = st.form_submit_button("Perform Analysis")

    if market_sentiment_analysis:
        # print (st.session_state.pydantic_class)

        if question:
            intermediate_answer_ag = AG()
            intermediate_answer_ag = asyncio.run(
                intermediate_answer_ag.generate_atype(question)
            )
            st.session_state.pydantic_class = intermediate_answer_ag.atype
            st.session_state.code = intermediate_answer_ag.atype_code

        with st.spinner(f"Performing {st.session_state.selected_model} ..."):
            start_index = selected_start
            end_index = selected_end

            sentiment = asyncio.run(
                (
                    AG(
                        atype=st.session_state.pydantic_class,
                        transduction_type="areduce",
                        areduce_batch_size=batch_size,
                    )
                    << st.session_state.dataset.filter_states(
                        start=start_index, end=end_index + 1
                    )
                )
            )
            sentiment.states += sentiment.areduce_batches
            sentiment = sentiment.add_attribute("question", default_value=question)
            answer = asyncio.run(
                AG(atype=Answer, transduction_type="areduce") << sentiment
            )
            import yaml

            col2.markdown(f"### Analysis from {selected_start} to {selected_end}")
            col2.text(
                yaml.dump(
                    sentiment[0].model_dump(), sort_keys=False, allow_unicode=True
                )
            )
            col1.text(
                yaml.dump(answer[0].model_dump(), sort_keys=False, allow_unicode=True)
            )

            col2.markdown("""### Intermediate Batches Analysis""")
            for state in sentiment.areduce_batches:
                col2.markdown("""""")
                col2.text(
                    yaml.dump(state.model_dump(), sort_keys=False, allow_unicode=True)
                )

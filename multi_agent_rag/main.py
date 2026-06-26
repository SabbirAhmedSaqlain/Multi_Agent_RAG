"""
Multi-Agent RAG System — Entry Point

Agents in the pipeline:
  1. RetrieverAgent   — vector search + relevance filtering
  2. AnalyzerAgent    — fact extraction & gap analysis
  3. SynthesizerAgent — coherent answer composition
  4. CriticAgent      — quality validation & revision
  5. OrchestratorAgent — coordinates all agents
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rag import DocumentStore
from agents import OrchestratorAgent


SAMPLE_DOCUMENTS = [
    {
        "source": "climate_science.txt",
        "content": """
Climate change refers to long-term shifts in global temperatures and weather patterns.
Since the 1800s, human activities have been the main driver of climate change,
primarily due to burning fossil fuels like coal, oil and gas.

Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped
around the Earth, trapping the sun's heat and raising temperatures.
Examples of greenhouse gas emissions that are causing climate change include carbon
dioxide and methane. These come from using gasoline for driving a car or coal for
heating a building, for example. Clearing land and forests can also release carbon dioxide.
Landfills for garbage are a major source of methane emissions.
Energy, industry, transport, buildings, agriculture and land use are among the main emitters.

The Intergovernmental Panel on Climate Change (IPCC), which includes more than 200 scientists
from 65 countries, concluded that limiting global temperature rise to 1.5°C above
pre-industrial levels would substantially reduce the impacts of climate change.
        """,
    },
    {
        "source": "renewable_energy.txt",
        "content": """
Renewable energy is energy derived from natural sources that are replenished at a higher
rate than they are consumed. Sunlight and wind, for example, are such sources that are
constantly being replenished.

Solar power is one of the fastest growing renewable energy technologies worldwide.
Solar photovoltaic (PV) technology converts sunlight directly into electricity using
semiconductor materials. The cost of solar PV has dropped by more than 80% over the last decade.

Wind energy is harnessed by wind turbines which convert kinetic energy from the wind into
mechanical power, which is then converted to electrical power using a generator.
Offshore wind farms have significantly higher wind speeds and capacity factors than onshore wind.

Hydropower is the most widely used renewable energy source globally. It generates electricity
by using the energy of flowing or falling water. Large hydropower plants can generate
thousands of megawatts of electricity.

By 2030, renewable energy could account for up to 60% of global electricity generation
according to the International Energy Agency (IEA), up from around 30% today.
        """,
    },
    {
        "source": "machine_learning.txt",
        "content": """
Machine learning is a subset of artificial intelligence that provides systems the ability
to automatically learn and improve from experience without being explicitly programmed.
Machine learning focuses on the development of computer programs that can access data and
use it to learn for themselves.

Deep learning is a subfield of machine learning that uses neural networks with many layers
(deep neural networks) to learn representations of data with multiple levels of abstraction.
Deep learning has been transformative for tasks like image recognition, natural language
processing, and speech recognition.

Large Language Models (LLMs) are neural networks trained on massive text datasets.
They have demonstrated remarkable abilities in text generation, translation, summarization,
question answering, and reasoning. Notable examples include GPT-4, Claude, and Gemini.

Transformer architecture, introduced in the 2017 paper "Attention Is All You Need",
is the foundational architecture behind most modern LLMs. It uses self-attention mechanisms
to process sequential data in parallel rather than sequentially.

Retrieval Augmented Generation (RAG) combines the power of LLMs with information retrieval.
It allows models to access external knowledge bases at inference time, improving accuracy
and reducing hallucinations. RAG is widely used for enterprise chatbots and question answering systems.
        """,
    },
    {
        "source": "space_exploration.txt",
        "content": """
Space exploration is the use of astronomy and space technology to explore outer space.
While the exploration of space is currently carried out mainly by astronomers with telescopes,
its physical exploration is conducted both by uncrewed robotic space probes and human spaceflight.

NASA's Artemis program aims to return humans to the Moon by 2026, including the first woman
and first person of color on the lunar surface. The program involves the Space Launch System
(SLS) rocket and the Orion spacecraft.

Mars is the most studied planet beyond Earth. The Perseverance rover, which landed in 2021,
has been collecting samples for future return to Earth and searching for signs of ancient
microbial life. It has also deployed the Ingenuity helicopter, the first powered aircraft
to fly on another planet.

SpaceX's Starship is the largest and most powerful rocket ever built, designed for missions
to the Moon and Mars. It uses a fully reusable design that could dramatically reduce the
cost of space access.

The James Webb Space Telescope (JWST), launched in December 2021, is the most powerful
space telescope ever built. It observes in infrared light and has provided unprecedented
views of galaxies, nebulae, and exoplanet atmospheres.
        """,
    },
]


def load_sample_data(store: DocumentStore) -> None:
    print("[Main] Loading sample documents...")
    for doc in SAMPLE_DOCUMENTS:
        doc_id = store.add_document(doc["content"], source=doc["source"])
        print(f"  Added '{doc['source']}' → {doc_id}")


def main():
    print("=" * 60)
    print(" Multi-Agent RAG System")
    print("=" * 60)

    # Initialize document store and load data
    store = DocumentStore()
    load_sample_data(store)
    print(f"\n[Main] Knowledge base: {store.stats()}")

    # Initialize the orchestrator (creates all sub-agents internally)
    orchestrator = OrchestratorAgent(store)
    orchestrator.build_index()

    # Run queries
    queries = [
        "How does RAG (Retrieval Augmented Generation) work and why is it useful?",
        "What are the main renewable energy sources and their growth potential?",
        "What is NASA's Artemis program and what are its goals?",
    ]

    for query in queries:
        result = orchestrator.run(query, verbose=True)

        print(f"\n{'='*60}")
        print("FINAL ANSWER:")
        print("=" * 60)
        print(result.final_answer)
        print(f"\n[Completed in {result.elapsed_seconds:.1f}s]")

        if not result.success:
            print(f"[ERROR]: {result.error}")

        print("\n" + "=" * 60 + "\n")

        # Pause between queries
        user_input = input("Press Enter for next query (or 'q' to quit): ").strip()
        if user_input.lower() == "q":
            break


if __name__ == "__main__":
    main()

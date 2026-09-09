# machine-learning-self-learning

## Context

Varun is learning Machine Learning as part of an enrolled course, alongside the parallel `python-self-learning`, `sql-self-learning`, and `statistics-self-learning` tracks in this same `2026` project folder. This is the track those three were building toward — Python for writing code, SQL for pulling/shaping data, and Statistics for the mathematical foundation ML sits on.

Chapter 1 ("Machine Learning Foundations") covers: why ML exists (vs. traditional rule-based programming), the three types of ML (Supervised, Unsupervised, Reinforcement) and their sub-types (Classification/Regression, Clustering/Dimensionality Reduction), a structured checklist for approaching a vague business problem and framing it as an ML problem, and a conceptual overview of each of the three learning types. Written to `01-Machine Learning Foundations/theory.md`, with hands-on scenario-based practice problems in `types_of_ml_practice.ipynb` — Varun asked for these filled in with working code + explanations (not scaffold-only) since this is early learning phase, each cell showing the *shape* of that ML type's data (labeled data for supervised, unlabeled for unsupervised, agent/action/reward for RL) with meaning + expected-output comments per line, same convention as `python-self-learning`'s reference notebooks.

## Working agreement

- Varun is new to ML — explanations should be basic, practical, and grounded in concrete real-world stories/examples, not just definitions or jargon.
- For every concept: explain **what it is**, **when/why you'd use it**, and **how to recognize it in a real scenario** — not just the textbook definition.
- Go one concept/topic at a time; do not dump multiple topics at once unless Varun asks for a full recap.
- Tie concepts back to the statistics foundation already covered in `statistics-self-learning` where relevant (e.g. categorical vs. numerical target → classification vs. regression; distributions/outliers mattering before training a model).
- Claude's role is to **explain and take notes together**, not just write finished notes unprompted going forward. Notes get written to `.md` files per chapter once a topic/chapter is actually covered/discussed.
- For hands-on practice notebooks: default is scaffold (scenario/question in markdown + an empty/guided code cell, Varun writes the code himself), matching `python-self-learning`'s graded-assignment convention — but Varun has said this ML track is early learning phase, so fill in working code with meaning + expected-output comments per line (matching `python-self-learning`'s reference-notebook convention) when asked, rather than leaving cells empty.
- Use small, realistic example scenarios (not abstract symbol-only explanations) when introducing a new concept.

## Environment

- Git identity for this repo is set locally to the personal GitHub identity (`varunprabhakaran22`), consistent with the other three self-learning repos.

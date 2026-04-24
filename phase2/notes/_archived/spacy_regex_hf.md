# prompt for implementing spacy regex and hugging face

Help me design a pipeline to extract and classify timeline dates from unstructured NEPA project text.

The text has already been extracted from PDFs into JSON or plain text. I want to identify all candidate dates, capture the surrounding context for each date, classify the role of each date in the NEPA process, and produce a structured table of results.

Please assume I may use spaCy, regex, and Hugging Face transformers.

I want the system to follow this logic:

1. Extract candidate dates using spaCy plus regex.
2. Normalize each date into a standard format when possible.
3. Capture context for each date, including sentence, nearby sentences, paragraph, section heading, and document type if available.
4. Classify each date into labels such as notice of intent, scoping start, draft release, comment period end, decision date, FONSI signed, ROD signed, construction start, or other.
5. Rank competing candidates and select the best project-level date for each timeline role.

Please give me:

* a recommended architecture
* the role of spaCy vs. Hugging Face
* whether I should begin rule-based, ML-based, or hybrid
* a suggested output schema
* pseudocode or Python scaffolding for implementation

Optimize for interpretability, scalability, and ease of debugging. Prefer a modular hybrid approach over a black-box end-to-end one unless there is a strong reason not to.

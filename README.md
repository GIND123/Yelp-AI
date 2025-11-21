# Yelp-AI: Image-Driven Query Generation and Multi-Agent Review Analysis

## Overview

This repository contains two independent yet interoperable AI pipelines integrating:
- Google Gemini 2.5 Flash Vision Models
- Yelp Fusion API (Business & Reviews)
- Yelp AI API (Natural-Language Search & Review Summaries)
- A full-scale multi-agent LLM debate architecture

The purpose is to build a complete technical system for:

1. Pipeline 1: Converting any user-provided image into a structured Yelp AI search query.
2. Pipeline 2: Performing a multi-agent (Optimist, Critic, Judge) debate-based evaluation of any Yelp business URL to derive pros, cons, and a verdict.

This README details methodologies, API interactions, execution workflows, and system architecture. It is written for engineers seeking a reproducible and technically rigorous pipeline.

---

# 1. Pipeline 1 — Image-Based Yelp Query Generation

File: Pipeline1.ipynb  
Objective: Generate a single well-formed Yelp AI API query based on:
- A user-provided image
- A free-text user query describing search intent
- Environmental variables: Location, Date, Time

The output is a structured natural-language query optimized for the Yelp AI API with strict constraints on tone, structure, and length.

## 1.1 Process Architecture

The system executes the following stages:

1. User inputs an image path and a natural-language question.
2. Image is encoded into binary form and provided to the Gemini 2.5 Flash Vision model.
3. A controlled instruction template guides the LLM to extract semantic details from the image and user query.
4. The model synthesizes a single natural-language query with explicit inclusion of Location, Date, and Time.
5. A post-processing length normalization stage ensures the query remains within 900 characters and ends at natural punctuation.
6. The final output is printed and may be used directly with the Yelp AI API.

## 1.2 Image Handling

Images are loaded as raw bytes and passed to Gemini as a vision-enabled data container. This enables feature extraction from food, products, clothing, interior décor, materials, colors, and other visual attributes.

## 1.3 Controlled Natural-Language Generation

A deterministic instruction template ensures:
- Exactly one query sentence.
- Explicit mention of Location, Date, and Time.
- Integration of both image-derived attributes and user intent.
- No references to the image, prompts, models, or APIs.
- A maximum allowable length.
- Strict compliance with Yelp-style user queries.

The instruction enforces the following template:

“Find <business/restaurant/store> in <location> that offer <item derived from the image> with <user-specified constraints>, available on <date> at <time>.”

## 1.4 Text Normalization

Outputs exceeding 1000 characters are truncated at the last punctuation mark to preserve grammatical integrity. This ensures model drift or excessive verbosity does not affect API compatibility.

## 1.5 Output

The final natural-language query is ready to:
- Be sent to the Yelp AI API.
- Be integrated into multi-turn conversational search agents.
- Serve as input for retrieval-based systems.

---

# 2. Pipeline 2 — Multi-Agent Review Debate System

File: Pipeline2.ipynb  
Objective: Accept a Yelp business URL and produce:
- An optimistic analysis
- A critical analysis
- A judge-synthesized verdict including Pros, Cons, and a final decision

This pipeline implements a full-scale three-agent LLM system using independent calls to ensure strict role separation and objective contrastive evaluation.

## 2.1 System Architecture

Pipeline 2 follows this structured sequence:

1. User inputs a Yelp business URL.
2. The system extracts the Yelp business ID from the URL.
3. The Yelp Fusion API is queried for business details (name, rating, price, categories, location).
4. The Yelp Fusion Reviews API attempts to fetch user reviews.
5. If direct reviews are unavailable (due to API plan restrictions), the Yelp AI API is used to generate synthetic review summaries.
6. A shared context block is constructed from:
   - Business metadata
   - Real review excerpts or AI-generated summaries
7. Three independent LLM calls are executed:
   - Optimistic Agent: highlights strengths.
   - Critical Agent: highlights weaknesses.
   - Judge Agent: receives both analyses plus the original context and synthesizes a neutral summary.
8. Judge produces exactly three lines:
   - Pros: <sentence>
   - Cons: <sentence>
   - Our verdict: <sentence>

This enforces deterministic formatting and consistent output structure.

## 2.2 Review Acquisition Strategy

### 2.2.1 Yelp Fusion Reviews API (Primary)

The system first attempts:
GET /v3/businesses/{business_id}/reviews  
This returns up to 20 recent review excerpts. Output is deterministic and directly tied to real Yelp user submissions.

### 2.2.2 Yelp AI API (Fallback)

If the above endpoint is unavailable due to plan restrictions, the system uses the Yelp AI API to generate a structured summary containing:
- Three positive points
- Three negative points

This ensures that the debate pipeline always receives curated review-like input, enabling agent-level analysis even without direct review access.

## 2.3 Multi-Agent Framework

Each agent executes as an independent LLM call. No role simulation occurs inside a single call. This ensures:
- Isolation of belief states
- Prevention of role leakage
- Higher reasoning diversity

### Optimistic Agent

Receives shared context. Produces 2–4 sentences emphasizing:
- Strengths
- Positive trends
- Desirable characteristics
- Areas contributing to customer satisfaction

### Critical Agent

Receives the same shared context but is instructed to emphasize:
- Weaknesses
- Recurring complaints
- Quality, service, or environment risks

### Judge Agent

Receives:
- Business context
- Optimistic analysis
- Critical analysis

Produces:
- Pros: <one sentence>
- Cons: <one sentence>
- Our verdict: <one sentence>

Sentences must be concise, under strict character limits, and avoid meta-discourse.

## 2.4 Context Consistency

The shared context passed to all three agents is identical and fixed. This ensures debate fairness and prevents inconsistent evidence pools.

## 2.5 Output Determinism

The judge’s final output:
- Must not mention the Yelp API or LLM mechanics.
- Must be purely business-facing.
- Must remain under character limits.
- Must follow exact formatting instructions.

---

# 3. Installation and Configuration

## 3.1 Required Python Packages

Install dependencies:

pip install google-genai  
pip install requests  
pip install urllib3

## 3.2 Environment Variables

Export API keys:

export API_KEY="YOUR_GOOGLE_GEMINI_KEY"  
export YELP_API_KEY="YOUR_YELP_API_KEY"

---

# 4. Running the Pipelines

## 4.1 Running Pipeline 1

Execute the notebook:

jupyter notebook Pipeline1.ipynb

The notebook will prompt for:
- Path to the image
- A user-specified natural-language query

Output is a structured Yelp AI search query.

## 4.2 Running Pipeline 2

Execute:

jupyter notebook Pipeline2.ipynb

The notebook will prompt for:
- A Yelp business URL

Output:
- Independent optimistic and critical agent analyses
- Final judge-derived pros, cons, and verdict

---

# 5. Extensibility

Both pipelines can be extended for:
- Multi-turn debate sequences
- Preference-weighted verdict generation
- Personalized recommendation logic
- Async parallel execution for reduced inference time
- Integration with conversational agents
- Automated reservation or ordering workflows via Yelp endpoints

---

# 6. License

MIT License.

---

# 7. Conclusion

This repository provides a modular and technically rigorous framework combining:
- Vision-based query generation
- Yelp Fusion and Yelp AI integration
- Full-scale multi-agent reasoning

The pipelines can serve as standalone components or integrated modules in recommendation engines, search assistants, and multimodal decision-support systems.

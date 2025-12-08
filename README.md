# WTF – Where’s The Food 🍔📸
**Track:** Mobile App

---

## Overview

**WTF (Where’s The Food)** is a mobile-first application that helps users identify where they can find a dish they see online or in real life.  
Users upload a screenshot or food photo, choose **location, date, and time**, and our system uses **computer vision + LLM reasoning + the Yelp AI API** to discover matching restaurants, rank options, and provide an **agent-driven dining verdict** — including whether it’s better to dine in or order delivery at that moment.

The app is designed for social-media-driven discovery:  
> *Saw food on Instagram or TikTok and want to know where to get it? Screenshot → upload → decide.*

---

## Core Features

### 📷 Food Image → Restaurant Search (Primary Yelp AI Workflow)
- Upload an image or provide a caption.
- Our AI generates a precise **Yelp AI query sentence** including:
  - Dish type inferred from the image
  - User intent (dietary preferences or style)
  - **Location, date, and time**
- Query is sent directly to **Yelp AI Chat API** to retrieve candidates.
- Results are ranked by **rating and review count** from Yelp’s data.

### 🗺️ Contextual Planning
Users specify:
- **Location**
- **Date**
- **Time**

This enables:
- Checking **availability patterns**
- Prioritizing places likely open and ready to serve
- Identifying ideal options for dine-in vs pick-up windows

---

### 🧠 Multi-Agent Dining Evaluation System

Each selected restaurant is analyzed through a **3-agent debate system:**

#### ✅ Optimistic Agent  
Summarizes:
- Strengths
- Food quality highlights
- Good service patterns
- Convenience and value

#### ❌ Critical Agent  
Identifies:
- Recurring drawbacks
- Reliability issues
- Crowding, cleanliness, or service risks

#### ⚖️ Judge Agent (Final Verdict)
Produces a **single neutral recommendation paragraph**:
- Balanced overall assessment
- Ideal visitor type or time window
- Cautions if relevant

The verdict answers:
> *Is this the right place for me right now? Order in or dine out?*

---

### 📞 Action Layer

Each recommendation includes instant actions:
- **📞 Call Now** – opens native phone dialer
- **🗓️ Book on Yelp** – deep links to Yelp’s reservation/booking page
- **📍 View Location** – quick navigation support

---

### 🛡️ Safety & Relevance Guardrails

A built-in moderation layer ensures:
- Only **food- or dining-related searches** proceed.
- Irrelevant or unsafe queries are blocked or redirected.
- Image uploads unrelated to dining discovery are automatically rejected.

This keeps the system aligned strictly with its intended use case.

---

## System Architecture


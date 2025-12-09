# WTF – Where’s The Food 🍔📸
**Track:** Mobile App

---


## Table of Contents

- [Overview](#overview)
- [How to Set-Up](#how-to-set-up)
- [Core Features](#core-features)
  - [📷 Food Image → Restaurant Search](#-food-image--restaurant-search-primary-yelp-ai-workflow)
  - [🗺️ Contextual Planning](#%EF%B8%8F-contextual-planning)
  - [🧠 Multi-Agent Dining Evaluation System](#-multi-agent-dining-evaluation-system)
  - [📞 Action Layer](#-action-layer)
  - [🛡️ Safety & Relevance Guardrails](#%EF%B8%8F-safety--relevance-guardrails)
- [System Architecture](#system-architecture)
- [Backend Pipelines](#backend-pipelines)
  - [🔹 Pipeline 1 – Image to Yelp Discovery](#-pipeline-1--image-to-yelp-discovery)
  - [🔹 Pipeline 2 – Multi-Agent Verdict System](#-pipeline-2--multi-agent-verdict-system)
- [Compliance with Hackathon Rules](#compliance-with-hackathon-rules)
- [Live App Builds](#live-app-builds)
- [App Screenshots](#app-screenshots)
- [Demonstration Video](#demonstration-video)


---

## Overview

**WTF (Where’s The Food)** is a mobile-first application that helps users identify where they can find a dish they see online or in real life.  
Users upload a screenshot or food photo, choose **location, date, and time**, and our system uses **computer vision + LLM reasoning + the Yelp AI API** to discover matching restaurants, rank options, and provide an **agent-driven dining verdict** — including whether it’s better to dine in or order delivery at that moment.

The app is designed for social-media-driven discovery:  
> *Saw food on Instagram or TikTok and want to know where to get it? Screenshot → upload → decide.*

---

## How to Set-Up

### 1\. Clone the Repository

Open your terminal and run the following command to clone the project into your desired folder:

```bash
git clone https://github.com/GIND123/Yelp-AI.git
```

### 2\. Navigate to the Project Directory

Move into the mobile application folder:

```bash
cd Yelp-AI/yelp-mobile
```

### 3\. Install Dependencies

Install the necessary packages using npm:

```bash
npm install
```

### 4\. Run the Application

Start the development server. This command will generate a QR code in your terminal:

```bash
npm start
```

### 5\. Launch on Your Device (Android)

To view the app on your physical device:

1.  Download and install the **Expo Go** app from the Google Play Store on your Android device.
2.  **Crucial:** Ensure your computer and your phone are connected to the **same Wi-Fi network**.
3.  Open Expo Go on your phone.
4.  Use the "Scan QR Code" feature within Expo Go to scan the QR code generated in your terminal.

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


---

## Backend Pipelines

### 🔹 Pipeline 1 – Image to Yelp Discovery
- Accepts uploaded images or text captions.
- Uses **Gemini multimodal generation** to produce a single precise Yelp query sentence.
- Queries **Yelp AI Chat Endpoint** to retrieve businesses.
- Normalizes:
  - Ratings
  - Review counts
  - Photos
  - Hours
  - Booking availability
- Sorts by rating + popularity.

> Implemented in: `Pipeline1Backend.py` :contentReference[oaicite:1]{index=1}

---

### 🔹 Pipeline 2 – Multi-Agent Verdict System

For a selected Yelp business:
1. Fetch business details and real reviews using **Yelp Fusion API**.
2. If reviews are unavailable, fallback to **Yelp AI summary extraction**.
3. Run the **Optimist, Critic, and Judge agents** using Gemini.
4. Produce the final actionable verdict.

> Implemented in: `Pipeline2Backend.py` :contentReference[oaicite:2]{index=2}

---

## Compliance with Hackathon Rules

✅ **Primary Data Source:** Yelp AI API  
✅ **No Third-Party Location Data Mix**  
✅ **Original Work Created During Submission Period**  
✅ **Fully Functional End-to-End Flow**  
✅ **Public Repository with Setup Instructions**  
✅ **Hosted Build for Testing**  
✅ **3-Minute Demo Video Included**

---

## Live App Builds

- **Android APK:**  
  👉 *[https://your-hosted-apk-link.com](https://expo.dev/accounts/rohithknair/projects/WTF/builds/16bd6259-00ad-4977-a6ca-5f413f3e06d0)*

---

## App Screenshots 

<img src="https://github.com/user-attachments/assets/37b5a10e-1164-4de9-87ed-84b230f91986" width="50%" height="50%">
<img src="https://github.com/user-attachments/assets/2562ea56-91e5-4fb2-8e88-3b47504825ae" width="50%" height="50%">
<img src="https://github.com/user-attachments/assets/e97f591c-37dc-49f6-8e5f-9d94b975e35a" width="50%" height="50%">

---

## Demonstration Video

🎥 Watch full demo (≈3 minutes):  
👉 *[https://youtube.com/your-demo-video](https://www.youtube.com/shorts/n8fyWitdWKs)*

The video covers:
- Image upload
- Time/location selection
- AI processing
- Yelp AI discovery flow
- Restaurant ranking
- Multi-agent verdict
- Booking/calling actions

---


Includes:
- Mobile frontend
- Backend pipelines
- Environment configuration
- API setup instructions

---



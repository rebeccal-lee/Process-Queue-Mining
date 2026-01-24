# Decision Support Tool for Emergency Rooms

> An analytics-driven platform for modeling, predicting, and optimizing Emergency Department (ED) patient flow using process mining, queueing theory, and machine learning.

---

## Overview

This project provides a decision-support system for ED triage leads and operations managers.  
It transforms raw patient event logs into actionable insights about bottlenecks, waiting times, patient outcomes, and resource utilization through an interactive dashboard.

The system supports:
- Process discovery of how patients actually move through the ED  
- Queue-level congestion and waiting-time analytics  
- Patient-level outcome and time-to-service prediction  
- Simulation of staffing and volume scenarios  
- Conformance analysis against expected clinical pathways  

---

## Data Flow

1. A raw ED event log is uploaded (arrival, triage, assessment, consult, discharge, etc.)  
2. The user maps their dataset’s column names to the program's data scheme
3. The system constructs an event log 
4. All analytics modules operate on the standardized patient journeys  
5. Results are visualized and explored through dashboards provided in the app. 

---

## Project Structure

### Home.py
- The main landing page for the application.  
- Handles file upload, column mapping, session state, and connects the dataset to all analytics modules.

### discovery.py 
Automatically generates a Directly-Follows Graph (DFG) from any uploaded CSV event log.  
Outputs transition frequencies and performance metrics per edge, designed to support interactive drill-down in the UI (e.g., filtering by triage level or zone). 

---

### conformance.py 
Implements basic conformance checking against a user-defined “Standard Protocol” (e.g., Triage → Registration → Assessment).  
Identifies and summarizes cases that deviate from the expected pathway (skipped steps, unexpected orderings, or detours).

---

### queue_mining.py 
Calculates and visualizes queue lengths and waiting times across shared resources (e.g., zones such as initial zone).  
Designed to surface operational bottlenecks impacting key KPIs like Time-to-PIA and LWBS risk. 

---

### predictive_analytics.py 
Builds features from partial patient journeys and predicts operational/clinical outcomes.  
Supports predictions such as probability of admission, probability of LWBS, and remaining time to PIA. 

---

### anomaly_detection.py 
Highlights “red flag” cases such as extremely long waits or unusual event sequences relative to baseline patterns.  
Used to surface outliers for operational review and escalation workflows.

---

### simulation.py 
Runs scenario-based ED flow simulation using a Bayesian time-to-event framework.  
It models service and delay durations with Weibull survival distributions, estimates parameters via MCMC, and propagates uncertainty through Monte Carlo sampling to simulate patient trajectories and quantify the impact of capacity changes on wait times and congestion. :contentReference[oaicite:7]{index=7}

---

### event_log_organizer.py 
Canonicalizes uploaded event logs into consistent, analysis-ready traces.  
Handles sequencing rules for simultaneous events, missingness handling, and normalization so discovery, conformance, queue mining, and ML operate on the same canonical event representation. :contentReference[oaicite:8]{index=8}

---

### load_data.py 
Loads raw event-log data and stores the user’s mapping for case ID, activity, timestamp, resource, and attributes.  
Produces a standardized schema compatible with all downstream modules and the dashboard

### pages/ Streamlit UI
- Contains the interactive dashboard pages.  
- Each file corresponds to a tab within the program (queue analysis, prediction, simulation, etc.) and generates interactive visual results.

---

## Use Case

This system is designed to support:
- Triage prioritization  
- Resource allocation  
- Bottleneck identification  
- LWBS risk management  
- Staffing and volume planning  

It enables ED leadership to move from static reporting to **data-driven operational decision making**.

---

## AI-Assisted Development Disclaimer

This project was developed with the assistance of generative AI tools (including large language models) used for code scaffolding, debugging support, refactoring, and documentation. All architectural design, algorithm selection, analytical methodology, data modeling, and validation decisions were made by the author.
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

---

### load_data.py
- Loads and standardizes raw event-log data.  
- Users map their dataset’s column names (case ID, activity, timestamp, resource, and attributes) and it is transformed into the program's data scheme.

---

### event_log_organizer.py
- Builds patient journeys from the standardized event log.  
- Applies ordering rules, step definitions, and state normalization to prepare data for analysis.

---

### discovery.py
- Performs process discovery on patient flows.  
- Constructs transition graphs, activity frequencies, and pathway statistics to reveal how patients actually move through the ED.

---

### queue_mining.py
- Implements queue-mining and waiting-time analytics.  
- Builds zone-level queues, computes queue lengths and wait distributions, and generates congestion and throughput visualizations.

---

### predictive_analytics.py
- Builds patient-level predictive models.  
- Uses partial event histories to predict outcomes such as admission risk, LWBS (left without being seen), and remaining time-to-physician.

---

### anomaly_detection.py
- Detects abnormal patient trajectories and delays.  
- Flags unusually long waits, rare pathways, and outlier cases relative to historical patterns.

---

### conformance.py
- Evaluates how observed patient flows deviate from expected or reference processes.  
- Quantifies skipped steps, delays, and off-path behavior for quality and compliance analysis.

---

### simulation.py
- Simulates ED operations under alternative demand and staffing scenarios.  
- Uses fitted arrival and service-time distributions to estimate how resource changes affect congestion and waiting times.

---

### pages/ (Streamlit UI)
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
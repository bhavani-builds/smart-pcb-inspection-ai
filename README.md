# 🔧 Smart PCB Inspection AI

An intelligent PCB inspection prototype that uses **Computer Vision, Image Processing, and Machine Learning** to detect, localize, and analyze potential PCB defects.

The system processes PCB images, identifies candidate abnormal regions, estimates their location and severity, and provides an interactive inspection dashboard.

---

## 🚀 Project Overview

Manual PCB inspection can be time-consuming and may be affected by human error.

This project demonstrates an automated computer-vision workflow for PCB inspection.

The system can:

- 📷 Process PCB images
- ✨ Enhance PCB images
- 🔍 Detect the PCB region
- 🧩 Analyze components and traces
- 🧪 Generate controlled defect examples
- 🔴 Detect candidate fault regions
- 📍 Localize detected regions
- ⚠️ Estimate defect severity
- 📊 Extract numerical image features
- 🤖 Experiment with ML-based fault classification
- 🖥️ Provide an interactive inspection dashboard
- 📄 Generate an inspection report

> **Important:** This is a computer-vision prototype. Detected regions are candidate visual anomalies and should not be treated as certified manufacturing defects.

---

# 🧠 System Architecture

```text
                    PCB IMAGE
                        │
                        ▼
              ┌──────────────────┐
              │ Image Enhancement│
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ PCB Region       │
              │ Detection        │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Component &      │
              │ Trace Analysis   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Defect Simulation│
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Fault Detection  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Fault             │
              │ Localization      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Feature           │
              │ Extraction        │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ ML Fault          │
              │ Classification    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Inspection        │
              │ Dashboard         │
              └────────┬─────────┘
                       │
                       ▼
              📄 INSPECTION REPORT

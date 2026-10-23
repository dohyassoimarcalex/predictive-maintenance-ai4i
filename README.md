# predictive-maintenance-ai4i
# ⚙️ PrediQ-Ops — Industrial Predictive Maintenance Engine

An end-to-end Machine Learning pipeline detecting critical machine failures before breakdown, balancing operational downtime costs against false maintenance alarms.

---

## 🎯 Project Summary
- **Pipeline & Physics:** We built a complete predictive maintenance pipeline using the AI4I dataset, incorporating physical variables such as mechanical power output and thermal gradients.
- **Handling Imbalance:** To address severe class imbalance, we prioritized high-recall metrics and a cost-sensitive random forest rather than relying on standard accuracy.
- **App Packaging:** The workflow was successfully transformed into an interactive Streamlit web application powered by a serialized model using `joblib`.
- **Deployment:** We versioned the repository on GitHub and configured cloud deployment, making the diagnostic tool fully operational and accessible.

---

## 📌 Business & Operational Context

In automated manufacturing lines, unplanned machine outages incur high costs per minute. Traditional fixed-threshold alerts either trigger too late or generate alert fatigue through false positives.

**PrediQ-Ops** leverages physical sensor telemetry to compute failure probabilities dynamically, flagging root causes before an unexpected stop occurs.

---

## 📊 Dataset & Physical Feature Engineering

* **Origin:** UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset).
* **Volume & Imbalance:** 10,000 observations, 3.4% failure rate.
* **Physics-Informed Features:**
  * **Mechanical Power Output ($P$):** $P = \tau \cdot \omega$ [W]
  * **Thermal Gradient ($\Delta T$):** $\Delta T = T_{\text{process}} - T_{\text{air}}$ [K]
  * **Mechanical Fatigue Index:** $\text{Torque} \times \text{Tool wear}$

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/dohyassoimarcalex/predictive-maintenance-ai4i.git](https://github.com/dohyassoimarcalex/predictive-maintenance-ai4i.git)
cd predictive-maintenance-ai4i
pip install -r requirements.txt
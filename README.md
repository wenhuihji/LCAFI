# LCFI: Label Correlation and Feature Interaction Framework for Imbalanced Label Distribution Learning (ILDL)

## Project Overview
This project implements the **LCFI** (Label Correlation and Feature Interaction) framework, designed to address the challenges of label correlation and feature interactions in Imbalanced Label Distribution Learning (ILDL). By explicitly modeling inter-label correlations and feature-label interactions, LCFI provides more accurate and stable learning results in applications with highly imbalanced label distributions.

The LCFI framework decouples the learning of dominant and non-dominant labels and applies a dynamic weighting mechanism to adjust their contribution during training. This helps mitigate the issue where high-degree labels dominate the training process, ensuring that low-degree labels also receive sufficient gradient feedback, thus improving model generalization and robustness.

## Key Features
- **Label Correlation Modeling**: Captures label dependencies through a self-attention mechanism, propagating semantic information from dominant to non-dominant labels.
- **Feature-Label Interaction**: Introduces label-guided feature embeddings to enhance semantic consistency between features and labels, helping improve learning for low-degree labels.
- **Label Distribution Decoupling and Dynamic Weighting**: Decouples the true label distribution into dominant and non-dominant branches and dynamically adjusts their optimization weights, addressing the bias caused by label imbalance.




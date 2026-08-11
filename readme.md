# LLM-GNN: Large Language Model Enhanced Graph Neural Network for User Identification

A hybrid framework that integrates semantic information from Large Language Models and structural information from Graph Neural Networks for social user identification.

## Overview

With the rapid growth of online social platforms, malicious users such as social bots and automated accounts have become increasingly prevalent. 
Traditional user identification methods usually rely on either textual information or network structures, which limits their ability to capture complex user behaviors.

In this project, we investigate a multimodal user identification framework by combining:

- **Language representation** extracted from pretrained language models
- **Graph representation** learned from user relationship graphs
- **Feature fusion mechanisms** for jointly modeling semantic and structural information

The goal of this project is to explore whether Large Language Models and Graph Neural Networks can complement each other for more effective user identification.

---

# Framework

The proposed framework consists of three major components:

             User Text
                |
              BERT
                |
      Text Representation
                |
                |
                +----------------+
                                 |
                              Fusion
                                 |
                +----------------+
                |
          Graph Representation
                |
          GNN Encoder
                |
           User Graph


The framework first extracts semantic embeddings from user-generated text using BERT, while Graph Neural Networks capture structural information from user relationships. Different fusion strategies are then explored to combine these two modalities.

---

# Datasets

## Cresci-RTBust-2019

Cresci-RTBust-2019 is a manually annotated social bot detection dataset containing both human users and bot accounts.

Characteristics:

- Real-world Twitter users
- Human/bot binary labels
- User-generated textual information
- Suitable for evaluating automated account detection methods

---

# Baseline Models

To evaluate the effectiveness of multimodal fusion, we first implement several single-modality baseline models.

## Text-based Baseline

### BERT Classifier

A pretrained BERT model is used to encode user-generated text and perform user classification.

The model focuses only on semantic information and ignores graph structures.

---

## Graph-based Baselines

Three Graph Neural Network baselines are implemented:

### 1. GCN

Graph Convolutional Network aggregates neighborhood information through graph convolution operations.

### 2. GAT

Graph Attention Network introduces attention mechanisms to dynamically assign importance weights to neighboring nodes.

### 3. GCN-Strong

An enhanced GCN baseline with additional optimization strategies for improving graph representation learning.

---

# Fusion Models

To investigate how textual and graph information can be effectively combined, six fusion strategies are designed and evaluated.

## Fusion-0: Basic Feature Fusion

Directly combines BERT embeddings with graph representations as node features.

---

## Fusion-1: Learnable Weighted Fusion

Introduces a learnable weight parameter to dynamically balance textual and graph information.

---

## Fusion-2: Attention-based Fusion

The main proposed fusion mechanism.

An attention module is introduced to adaptively learn the contribution of different modalities.

---

## Fusion-3: Feature-level Attention Fusion

Applies attention mechanisms at the feature representation level.

---

## Fusion-4: Confidence Gate Fusion

Uses a gating mechanism to control information flow between text and graph representations.

---

## Fusion-5: Disagreement-aware Fusion

Considers potential conflicts between textual and structural information and adjusts fusion strategies accordingly.

---

# Experimental Pipeline

The complete experimental pipeline is:

             Dataset

                |
    +-----------+-----------+
    |                       |
 Text Data              User Graph
    |                       |
   BERT                   GNN
    |                       |
    +-----------+-----------+
                |
          Fusion Models
                |
          Classifier
                |
        Human / Bot Prediction


---

# Experimental Results

## Baseline Comparison

| Model | Modality | Accuracy |
|------|----------|----------|
| BERT | Text | - |
| GCN | Graph | - |
| GAT | Graph | - |
| GCN-Strong | Graph | - |

---

## Fusion Comparison

| Model | Fusion Strategy | Accuracy |
|------|-----------------|----------|
| Fusion-0 | Basic Fusion | - |
| Fusion-1 | Weighted Fusion | - |
| Fusion-2 | Attention Fusion | **Best** |
| Fusion-3 | Feature Attention | - |
| Fusion-4 | Confidence Gate | - |
| Fusion-5 | Disagreement-aware Fusion | - |

---

# Repository Structure

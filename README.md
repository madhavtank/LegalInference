# Document-Level Natural Language Inference in Contracts (ContractNLI)

## **Overview**
ContractNLI is a project designed to address the challenge of Natural Language Inference (NLI) in legal contracts. Contracts often contain dense, complex language, requiring accurate classification of relationships between a contract and a hypothesis, as well as identification of specific evidence supporting these classifications. 

**Tasks**:
1. **NLI Classification**: Classify whether a hypothesis is:
   - *Entailed* by the contract,  
   - *Contradicted* by the contract, or  
   - *Not Mentioned* (neutral).  
2. **Evidence Identification**: Identify specific spans within the contract that justify the NLI decision.

This project provides scalable methods to reduce manual effort in industries like finance, law, and insurance, where efficient contract analysis is crucial.

---

#### **Exploratory Data Analysis (EDA)**
We analyzed the dataset to gain insights for model development:
- **Data Structure**: Contracts in JSON format, containing metadata, text, spans, and annotation sets.  
- **Key Findings**:
  - Label distribution: 50% entailment, 15% contradiction, 35% not mentioned.
  - Evidence spans: Median span length is 30 tokens; average of 10 spans per document.
  - Span variability: Lengths and densities vary significantly, requiring flexible models.

---

## **Implemented Approaches**
We developed and compared several models to handle the tasks:

### **1. Baseline Models**
- **Whole Document NLI**: Applied models like RoBERTa and DeBERTa on entire documents, truncating input at 512 tokens due to transformer limitations.  
- **Span TF-IDF**:
  - **SVM**: Converts spans and hypotheses into TF-IDF vectors, concatenates them, and uses an SVM classifier to predict evidence relevance.  
  - **Cosine Similarity**: Computes cosine similarity between TF-IDF vectors of spans and hypotheses to rank spans.  
- **Results**:
  - The SVM-based model outperformed cosine similarity, with higher precision at 80% recall (P@R80) and mean average precision (mAP).  

---

### **2. Span-Based Transformer Models**
- **RoBERTa/DeBERTa for NLI task**:
  - Reformulated inputs to pair spans with hypotheses (`Span [SEP] Hypothesis`).  
  - Focused on relevant document sections instead of full-document truncation.
  - By training on spans, DeBERTa achieved near-perfect results, significantly outperforming its performance in the full-document approach. Similarly, RoBERTa also showed substantial improvement, reinforcing the effectiveness of focusing on spans for the Natural Language Inference task.  
- **Separate Entailment and Contradiction Models**:
  - Trained two BERT-based models independently for entailment and contradiction.  
  - Enhanced evidence identification by specializing in each task.  

---

### **3. Advanced SpanNLI-BERT**
- **Dynamic Context Segmentation**:
  - Divided long documents into overlapping contexts to fit transformer input limits.  
  - Ensured spans remained intact with sufficient surrounding context.  
- **Model Features**:
  - Two MLPs on top of BERT:
    - One predicts NLI labels (entailment, contradiction, or neutral) using the `[CLS]` token.  
    - The other identifies evidence spans using `[SPAN]` tokens.  
  - Combines predictions to achieve state-of-the-art performance.  
- **Results**:
  - P@R80: 0.6539 | mAP: 0.8701 | Accuracy: 82.71%.  

---

### **4. Additional Approaches**
- **LSTM**:
  - Processes full contracts without truncation but underperforms transformers in precision and F1 score.  
- **Retrieval-Augmented Generation (RAG)**:
  - Combines semantic search for span retrieval with BERT-based NLI classification.  
  - Effective for scaling to large datasets but dependent on retrieval quality.

---

#### **Evaluation Metrics**
Key metrics used to evaluate models:
- **Accuracy**: Measures overall correctness.  
- **Precision @ 80% Recall (P@R80)**: Assesses precision when recall is at least 80%.  
- **Mean Average Precision (mAP)**: Evaluates ranking performance of spans across recall levels.

---

#### **Results Summary**
| **Model**            | **P@R80** | **mAP**   | **Accuracy** |  
|-----------------------|-----------|-----------|--------------|  
| Span TF-IDF + SVM    | 0.3152    | 0.4157    | N/A          |  
| Span based DeBERTa   | N/A       | N/A       | 0.9498       |
| SpanNLI-BERT         | 0.6539    | 0.8701    | 82.71%       |  

SpanNLI-BERT emerges as the most effective, particularly in tasks requiring fine-grained evidence identification and dense document understanding.

(More detailed explanation of the approaches and results can be found in the report)

---

#### **Contributors**
- **Manav Shah**, **Chirag Jain**, **Madhav Tank**
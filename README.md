# 📝 Neet OMR sheet scanner/grader and rank predictor
 
A Python-based web application built with **Flask, OpenCV, Pillow, and Scikit-learn** for automating the evaluation of NEET OMR sheets.  
It can process **up to 800 bubbles** with high accuracy, calculate scores using the official NEET grading convention, and even **predict a candidate’s rank** with polynomial regression.  

---

## 🚀 Features  
- Accepts **two input images**:  
  - **Answer Key OMR Sheet**  
  - **Student OMR Sheet**  
- Preprocessing with **Pillow + OpenCV**:  
  - Convert to binary  
  - Remove noise  
  - Clarify bubbles  
- **Edge detection & cropping** to align both sheets  
- **Homography** with **SIFT / ORB / SURF** and **800 manually set template points**  
- Accurate **bubble detection** for marked responses  
- Automatic **scoring** with NEET marking scheme (+4 / –1 / 0)  
- **Rank prediction** using **Polynomial Regression** (Scikit-learn)  
- **Visualization** of score vs. rank using **Matplotlib**  

---

## ⚙️ How It Works  

1. **Input**  
   - Upload two scanned OMR sheets: the **Answer Key** and the **Student’s sheet**.  

2. **Preprocessing**  
   - Images are converted to **binary** for clarity.  
   - Noise is removed and edges are enhanced.  
   - Both sheets are **cropped** to focus on the bubble area.  

3. **Alignment (Homography)**  
   - Keypoints are extracted using **SIFT / ORB / SURF**.  
   - A **homography matrix** aligns the student sheet with the template.  
   - Mapping is done to **800 predefined bubble positions**.  

4. **Bubble Detection**  
   - Each bubble region is analyzed to check if it’s **filled or empty**.  

5. **Scoring**  
   - Student responses are matched against the answer key.  
   - Marks are assigned according to **NEET rules**:  
     - ✅ Correct → +4  
     - ❌ Incorrect → –1  
     - ⭕ Unattempted → 0  

6. **Rank Prediction**  
   - **Polynomial Regression** estimates the candidate’s rank from the score.  
   - A **score vs. rank curve** is plotted with **Matplotlib**.  

7. **Results**  
   - Candidate’s total score  
   - Question-wise evaluation  
   - Predicted NEET rank  

---

## 🛠️ Tech Stack  
- **Python** – core logic  
- **Flask** – web framework  
- **OpenCV** – preprocessing, edge detection, feature matching, homography  
- **Pillow** – image filtering and cleanup  
- **Scikit-learn** – polynomial regression for rank prediction  
- **Matplotlib** – data visualization  

---

## 📌 Project Highlights  
- Combines **computer vision** and **machine learning** in one tool  
- Handles large-scale bubble detection (**800 bubbles**) efficiently  
- Provides not only scores but also **insightful rank predictions**  
- End-to-end automated evaluation with a **user-friendly interface**  

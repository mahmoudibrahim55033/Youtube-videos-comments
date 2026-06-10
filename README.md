# 💬 YouTube Comments Clustering (NLP + K-Means)

An unsupervised NLP project that groups YouTube comments into thematic clusters using TF-IDF vectorization and K-Means clustering. The project covers the full pipeline — from raw comment text cleaning and feature extraction to cluster assignment and distribution visualization.

---

## 📁 Dataset

- **File:** `YT_Videos_Comments.csv`
- **Source:** [Kaggle — YouTube Videos and the Comments](https://www.kaggle.com/datasets/japkeeratsingh/youtube-videos-and-the-comments)
- **Rows:** ~3.2 million comments
- **Original columns:** 9 (reduced to 2 for modeling)

| Column | Description |
|---|---|
| `User` | YouTube channel/user associated with the video |
| `Video Title` | Title of the video (dropped) |
| `Video Description` | Description of the video (dropped) |
| `Video ID` | Unique video identifier (dropped) |
| `Comment (Displayed)` | Rendered comment text (dropped) |
| `Comment (Actual)` | Raw comment text — **used for NLP** |
| `Comment Author` | Username of the commenter (dropped) |
| `Comment Author Channel ID` | Channel ID of the commenter (dropped) |
| `Comment Time` | Timestamp of the comment (dropped) |

---

## 🔍 Project Workflow

### 1. Data Cleaning
- Dropped 7 non-informative columns, retaining only `User` and `Comment (Actual)`
- Filled missing comments with the mode value

### 2. Text Preprocessing
Each comment was cleaned through the following steps:

1. **Remove URLs** using regex (`http`, `www` patterns)
2. **Remove non-alphabetic characters**
3. **Lowercase** all text
4. **Tokenize** by splitting on whitespace
5. **Remove stopwords** using NLTK's English list — with `"not"` kept to preserve negation
6. **Stemming** using Porter Stemmer

### 3. Feature Extraction — TF-IDF
- `TfidfVectorizer` with `max_features=1000`
- Produces a weighted matrix where rare but informative words score higher than common ones
- Output cast to `float16` to reduce memory usage on the large dataset
- `StandardScaler` applied before clustering

### 4. Optimal Cluster Selection
- Tested K from 1 to 10 using the **Elbow Method** (inertia plot)
- **Chosen K: 3**

### 5. Final Model & Visualization
- K-Means with `k=3`, `init='k-means++'`, `max_iter=300`, `n_init=10`
- Cluster labels assigned back to the DataFrame
- **Count plot** showing the distribution of comments across the 3 clusters

---

## 📊 Evaluation

| Method | Purpose |
|---|---|
| Inertia (Elbow) | Identifies the optimal number of clusters |
| Cluster Distribution Plot | Shows balance/imbalance across comment groups |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, kagglehub

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mahmoudibrahim55033/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK stopwords (runs automatically in the notebook, or manually):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook yt-comments.ipynb
   ```

> The dataset is downloaded automatically via `kagglehub`. You'll need a Kaggle account and API key configured, or place `YT_Videos_Comments.csv` manually in the working directory and update the file path in the notebook.

> ⚠️ The dataset contains ~3.2 million rows. Processing may take several minutes depending on your hardware.

---

## 👤 Author

**mahmoudibrahim55033** — [GitHub Profile](https://github.com/mahmoudibrahim55033)

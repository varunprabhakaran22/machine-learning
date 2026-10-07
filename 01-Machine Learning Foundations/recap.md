# Recap — Chapter 1: Machine Learning Foundations

> 💬 = points added from doubt-clearing sessions (not in `theory.md`)

## 0. Why ML?

```
Traditional:  Rules + Data    → Program → Answers
ML:           Data + Answers  → Program → Rules (the "model")
```

- **ML** = software that learns patterns from data instead of hand-written `IF/ELSE`.
- Exploded recently because of **Data + Compute (GPUs) + better Algorithms**.
- **Reach for ML only when:** rules are too complex/unknown, patterns change over time (spam, fraud), and you have enough historical data. If a simple rule works → no ML.

---

## 1. The 3 Types (decided by: *what feedback does it learn from?*)

| Type | Labels? | Does what | Sub-tasks | Example |
|---|---|---|---|---|
| **Supervised** | ✅ Yes | Predicts a known kind of answer | Classification, Regression | Spam, house price |
| **Unsupervised** | ❌ No | Discovers structure | Clustering, Dimensionality Reduction | Customer segments |
| **Reinforcement** | ❌ No (gets reward/penalty) | Learns a strategy by trial & error | — | Games, self-driving, robots |

💬 **Only 3 types exist.** Clustering / Dimensionality Reduction / Classification / Regression are **tasks inside** a type, not types themselves.

---

## 2. Supervised — "learning with an answer key"

- Story: studying from **solved past papers**, then answering a new exam.
- **Pick the sub-type by the target:**
  - Category (Nominal/Ordinal) → **Classification** (spam? churn yes/no?)
  - Number (Discrete/Continuous) → **Regression** (price, spend next month)
- ⚠️ Stats "discrete" = countable numbers → **Regression**, not Classification.
- Workflow: collect labeled data → train/test split → `fit()` → evaluate on test → deploy.
- Most real business ML is supervised.

---

## 3. Unsupervised — "learning without an answer key"

- Story: box of uncaptioned photos → you group similar ones yourself.
- **Clustering** → groups similar **rows** (K-Means). Human names the groups afterwards.
- **Dimensionality Reduction** → shrinks **columns**, e.g. 500 → 10 (PCA).

💬 **Dimensionality Reduction ≠ data cleaning**
| Data cleaning | Dimensionality reduction |
|---|---|
| Fixes bad data (nulls, dupes) with rules you write | **Learns** patterns in good data (`pca.fit(X)`) |
| Nothing learned | Discovers that columns move together (milk + bread + eggs → "grocery habit") |

It's "unsupervised" because it never looks at a label column. **Unsupervised ≠ clustering**; it means *learning without labels*.

💬 **Does unsupervised predict?** It never predicts a *known answer*. But a trained model can still **assign** a new row ("new customer → Cluster 2"). That's similarity-based, with no ground truth to verify.
👉 *Supervised predicts an answer; unsupervised describes structure.*

💬 **The Unsupervised → Supervised map (supermarket example)**
```
Raw customers (no labels)
   → Clustering → segments ("Budget", "Premium"...)
   → Business question needs an answer column → SUPERVISED
        ├─ Category? (will churn?)        → Classification
        └─ Number?   (spend next month?)  → Regression
```
- Connect them by: **cluster as a feature**, or **one supervised model per segment**.
- Unsupervised-first is **not mandatory**. Already have labels? Go straight to supervised.

---

## 4. Reinforcement — "learning by trial and error"

- Story: learning to ride a bicycle (fall = penalty, balance = reward).
- **Agent** (learner) → **Action** → **Environment** (world) → **Reward/Penalty + new state** → repeat millions of times → learns a **Policy** ("in this situation, do this").
- Harder than others: feedback is **delayed**, actions change future data, needs huge trial & error (usually in a **simulator**).
- Use RL only when the problem is a **sequence of decisions** scored by reward, not labels.

💬 **What a "penalty" is in code:** just a **negative number** (reward = positive). The agent is built to **maximize total score**.

💬 **Who gives the reward? Not the agent.** Engineers write a **reward function**; the environment runs it after each action.
```js
function getReward(s) {            // written by HUMANS
  if (s.crashed)            return -1000;
  if (s.redLightJumped)     return -500;
  if (s.reachedDestination) return +1000;
  return s.distanceProgressed * 1 - Math.abs(s.jerk) * 0.1;
}
// loop: action → simulator.step() → getReward() → agent.learn()
```
- Learning (Q-learning idea): a score table like `Q[state][action]`. Each reward nudges the score; the agent picks the highest-scoring action. Penalties push bad actions' scores down.
- ⚠️ **Reward hacking:** a bad reward function gets exploited (reward only "no crash" → car never moves). Designing the reward is the hardest part.

---

## 5. Parametric vs Non-Parametric (a *second, independent* axis)

Question: *once trained, how does the model store what it learned?*

| | Parametric | Non-Parametric |
|---|---|---|
| Stores | Fixed few numbers (e.g. slope + intercept) | Keeps (some or all) training data |
| Size vs data | **Fixed** | **Grows** with data |
| Shape assumption | Assumes one (e.g. straight line) | None, data decides |
| Prediction speed | Fast | Slower (KNN searches data) |
| Data needed | Less | More |
| Risk | **Underfit** (too simple) | **Overfit** (memorizes noise) |
| Interpretability | Easy (read the formula) | Harder |
| Examples | Linear/Logistic Regression, Naive Bayes, fixed Neural Nets | KNN, Decision Trees, Random Forest, SVM (kernels) |

- Story: Student A distills 1000 questions into a 10-formula cheat sheet; Student B keeps all 1000 and finds similar ones.
- House example (1350 sqft): Linear → `9.7 + 0.040×sqft ≈ 64.2L`; KNN (k=2) → avg(58, 71) = 64.5L.
- **One-question test:** *"Give it 10× more data. Does the model grow?"* No → Parametric. Yes → Non-Parametric.

💬 **Who decides it?** Neither `fit()` nor the model. It's a **fixed property of the algorithm**. Picking `LinearRegression()` means parametric, always.

💬 **Why should I care then?** Because it guides **which algorithm you pick before `fit()`**: model file size (`.pkl`), prediction speed (browser/mobile?), how much data you have, how complex the pattern is, whether you must explain it.

💬 **Can I customize it?** You can't change the category, but you can control behavior with **hyperparameters** (`max_depth=3`, `n_neighbors=5`).
- **Parameters** = learned by `fit()` (e.g. `coef_`). You don't set them.
- **Hyperparameters** = settings **you** choose before `fit()` (tuned in Ch. 9).

---

## How to approach a real business problem (checklist)

1. **Is it even ML?** Would a simple rule/SQL work reliably?
2. **Which box?** Supervised / Unsupervised / RL (then Classification or Regression).
3. **Do I have the data** (with labels, if supervised)?
4. **Define success measurably**, upfront ("catch ≥90% fraud, <2% false flags").
5. **Cost of being wrong?** Costly → keep a human in the loop.
6. **Start simple** (e.g. Logistic Regression) as a baseline; only go fancy if it clearly wins.

---

## 🚀 Roadmap links to GenAI / Agentic AI (revisit later)

- **Dimensionality Reduction → Embeddings:** compressing data into a few meaningful numbers is what embeddings do (text → vector). That's the foundation of vector DBs and RAG.
- **RL → RLHF:** ChatGPT/Claude are fine-tuned with RL from Human Feedback. Humans rate answers, a reward model learns preferences, and the LLM maximizes that reward.
  - 💬 **Your 👍/👎 click doesn't train the live model.** The deployed model is **frozen**. Clicks are logged → millions collected → an **offline** training run → a new version is deployed (like analytics → next release).
  - 💬 **Parameter count never changes** (70B stays 70B). Training only changes their **values**, the same way Linear Regression keeps 2 params while their values update.
  - 💬 **Chat "memory" isn't learning.** It saves text and pastes it into the next prompt; the parameters are untouched. **Changing parameters** = training (slow, offline). **Changing context** = prompts/memory/RAG (instant, runtime). Agent work is mostly the second.
- **Parametric → LLMs:** "7B / 70B" = billions of **parameters**. LLMs are parametric (fixed size, training data not stored), so they can't look up new facts. **RAG** adds a lookup at prediction time to fix that.

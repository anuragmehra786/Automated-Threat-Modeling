# Project Architecture: Automated Threat Modeling & Attack Path Mapper

## 1. What are we actually building?

Your project is Automated Threat Modeling.

At a high level:

```text
                    USER
                     │
                     │ enters authorized URL
                     ▼
              ┌──────────────┐
              │   FRONTEND   │
              │  Member 3    │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   BACKEND    │
              │  Member 2    │
              └──────┬───────┘
                     │
                     ▼
           ┌─────────────────────┐
           │ WEB DISCOVERY       │
           │ Member 2            │
           └─────────┬───────────┘
                     │
                     │ DiscoveryResult
                     ▼
           ┌─────────────────────┐
           │ SECURITY INTELLIGENCE│
           │ YOU                 │
           ├─────────────────────┤
           │ Asset Intelligence │
           │ Threat Modeling     │
           │ OWASP/CWE/MITRE    │
           │ RAG                 │
           │ LLM Reasoning       │
           │ Risk Engine         │
           │ Attack Paths        │
           └─────────┬───────────┘
                     │
                     │ SecurityAnalysis
                     ▼
              ┌──────────────┐
              │   BACKEND    │
              │  Member 2    │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   FRONTEND   │
              │  Member 3    │
              └──────────────┘
```

So the project is essentially:
Discover → Understand → Model threats → Assess risk → Explain → Visualize

---

## 2. The three members are building three major layers

Think of the project as a factory.

**Member 2 = Eyes + delivery system**
Member 2 discovers what can be observed from the authorized website and provides that information to your cybersecurity layer.

**You = Brain**
You take the observations and perform the actual cybersecurity intelligence:
* What are the assets?
* What threats might affect them?
* What security frameworks apply?
* What is the likely risk?
* What are possible attack paths?
* What mitigations should be recommended?
* What does the evidence actually support?

**Member 3 = Interface**
Member 3 takes your final analysis and makes it understandable to a human:
* dashboard
* risk score
* threats
* assets
* attack paths
* recommendations
* reports

This separation is extremely important.

---

## 3. COMPLETE PROJECT FLOW

Let’s imagine the user enters: `https://example.com`

The complete system eventually works like this.

### STEP 1 — User enters URL

Member 3 creates the interface. For example:

```text
┌──────────────────────────────────────────────┐
│        AUTOMATED THREAT MODELING             │
│                                              │
│ Target URL                                   │
│ ┌──────────────────────────────────────────┐ │
│ │ https://example.com                      │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ Protection priorities                        │
│ ☑ Authentication                            │
│ ☑ Data exposure                             │
│ ☑ Web application                           │
│                                              │
│             [ ANALYZE ]                      │
└──────────────────────────────────────────────┘
```

This is Member 3’s work. They don’t need to understand how threat modeling works internally. They just need to send the appropriate request to the backend.

---

### STEP 2 — Backend receives the request

This is primarily Member 2’s responsibility.

For example:
`POST /analyze` with:
```json
{
  "url": "https://example.com"
}
```

The backend:
1. receives the URL
2. validates it
3. creates an analysis job/request
4. calls the discovery module
5. obtains DiscoveryResult
6. passes that result to your security-analysis pipeline
7. receives SecurityAnalysis
8. returns/stores the result for the frontend

So Member 2 is essentially creating the orchestration/delivery layer.

---

### STEP 3 — Web Discovery

This is also Member 2’s major responsibility.

Their job is NOT to determine whether the website is vulnerable. Their job is:
“What can we safely and legitimately observe about this target?”

For an authorized target, their discovery module could observe things such as:
- **Pages**: `/`, `/login`, `/register`, `/about`, `/admin`
- **Links**: `/login → /register`, `/login → /reset-password`
- **Forms**: `/login` (username, password)
- **Endpoints**: `GET /`, `POST /api/login`, `GET /api/users`
- **Technologies**: React, Express, Nginx
- **HTTP/security indicators**: HTTPS = true, HSTS = present, CSP = absent, Secure cookie = true, HttpOnly = true, SameSite = Lax

---

### 6. What Member 2 should NOT do

This distinction needs to be extremely clear to them.

Member 2 should not build:
❌ AI threat reasoning
❌ RAG
❌ OWASP reasoning
❌ CWE reasoning
❌ MITRE reasoning
❌ LLM analysis
❌ risk calculation
❌ attack-path intelligence
❌ vulnerability exploitation

Their job ends roughly here:
Authorized URL ↓ Safe discovery ↓ Structured observations ↓ DiscoveryResult

---

### 7. The most important output from Member 2

Their entire discovery system eventually needs to produce something conceptually like:

```json
{
  "url": "https://example.com",
  "technologies": [...],
  "pages": [...],
  "endpoints": [...],
  "forms": [...],
  "security_indicators": {...}
}
```

That is the contract between Member 2 and you.
And we’ve already created the Pydantic models for this in: `app/models/schemas.py`

So Member 2’s AI system should build their discovery code against this contract, rather than inventing an entirely different data format.

---

### STEP 4 — Your Asset Intelligence

Now Member 2’s work enters your territory.
Suppose Member 2 discovers: `POST /api/login` and `/login` with `username` and `password`.
You don’t simply copy those into a list.

Your security layer asks: *What security-relevant asset does this evidence represent?*

For example:
Observed: `POST /api/login`
Observed: login form contains password field
↓
Inferred asset: Authentication Service

Your system could produce:
```json
{
  "name": "User Authentication Service",
  "type": "authentication_service",
  "status": "INFERRED",
  "confidence": 0.90,
  "evidence": [
    "POST /api/login",
    "Login form with password field"
  ]
}
```

---

### 9. Why this is more than a scanner

A normal scanner might say:
Found: `/login`, `/api/login`, Express, React

Your system should eventually say:
- Security-relevant assets: Authentication Service, Public Login Endpoint, Session Management, Administrative Interface
- Potential security concerns: Authentication abuse, Session-related threats, Access-control concerns, Client-side injection defenses
- Risk: ...
- Evidence: ...
- Confidence: ...

That’s where your cybersecurity contribution becomes significant.

---

### STEP 5 — Threat Modeling

After assets are identified, your system asks: *What threats could affect these assets?*

For example:
Asset: Authentication Service
Potential threats:
├── Spoofing
├── Credential attacks
├── Authentication abuse
└── Account takeover scenarios

But remember: Potential threat ≠ confirmed vulnerability.
If we observe `POST /login`, we can reasonably identify: Potential authentication abuse threat.
But we cannot automatically claim: ❌ "The website is vulnerable to brute force." because we haven’t verified its backend controls.

---

### 11. STRIDE

Your threat-modeling layer can use STRIDE as one of its conceptual frameworks.
S — Spoofing
T — Tampering
R — Repudiation
I — Information Disclosure
D — Denial of Service
E — Elevation of Privilege

For example, Asset: Admin interface → Potential threats: Spoofing, Elevation of Privilege, Information Disclosure.

---

### STEP 6 — Security Knowledge Base

Your project shouldn’t rely entirely on the LLM’s memory.
You build a cybersecurity knowledge base containing authoritative information.

Conceptually:
```
knowledge_base/
    owasp/
    cwe/
    mitre/
```

The knowledge might contain:
- OWASP A07: Authentication failures
- CWE-307: Improper Restriction of Excessive Authentication Attempts
- MITRE T1110: Brute Force

---

### STEP 7 — RAG

Your pipeline will eventually look like:
Security Documents ↓ Cleaning ↓ Chunking ↓ Metadata ↓ Embeddings ↓ FAISS ↓ Vector Search ↓ Relevant Security Knowledge

Suppose your asset layer identifies: Authentication Service
and your threat layer identifies: Potential authentication abuse
The RAG system can retrieve relevant knowledge (OWASP A07, CWE-307, MITRE T1110) and provide that context to your AI reasoning layer.

---

### STEP 8 — LLM reasoning

Now your AI layer gets something like:
Discovery evidence + Security assets + Potential threats + Retrieved security knowledge
↓ LLM ↓ Security reasoning ↓ Structured output

The LLM can help answer:
* Why is this asset security-relevant?
* Which potential threats are relevant?
* What evidence supports them?
* Which OWASP/CWE/MITRE mappings are appropriate?
* What assumptions are being made?
* What should remain UNKNOWN?
* What mitigation should be recommended?

But the LLM should not control everything.

---

### 15. LLM vs deterministic Python

This distinction is very important.

**LLM:** Good for: Interpretation, Reasoning, Classification, Explanation, Framework mapping, Recommendations
**Python:** Good for: Validation, Risk calculation, Score calculation, Severity calculation, Data consistency, Deduplication, Graph construction

For example:
LLM says: Likelihood = 4, Impact = 4
↓ Python: 4 × 4 = 16 ↓ HIGH
This makes the system more reliable.

---

### STEP 9 — Risk Engine

Your deterministic risk engine then calculates: Risk = Likelihood × Impact
For example:
Likelihood = 4, Impact = 4 → 4 × 4 = 16 = HIGH

The LLM doesn’t get to say: "I think this is 23." Your Python rules decide the final score.

---

### STEP 10 — Attack-path intelligence

Instead of only showing isolated findings (Threat A, Threat B, Threat C), your system can reason about relationships.

For example:
Internet ↓ Public Login Endpoint ↓ Authentication Service ↓ User Account
This becomes a potential attack path. Again: Potential attack path ≠ proof that the attack was successfully performed.
You can use NetworkX or another graph representation in your security layer.

---

### STEP 11 — Final SecurityAnalysis

Eventually your entire cybersecurity layer produces: `SecurityAnalysis`

which contains:
Summary, Assets, Threats, Risks, Attack Paths, Recommendations

Conceptually:
```json
{
  "summary": {...},
  "assets": [...],
  "threats": [...],
  "risks": [...],
  "attack_paths": [...],
  "recommendations": [...]
}
```
This is your main output contract to Member 3.

---

### STEP 12 — Member 3 visualizes it

Now Member 3 receives your SecurityAnalysis. They don’t need to calculate cybersecurity risk. They display it.

For example:
```text
┌──────────────────────────────────────────────┐
│ SECURITY ANALYSIS                            │
│                                              │
│ Overall Risk: HIGH                           │
│                                              │
│ Assets                 5                     │
│ Potential Threats     8                     │
│ Critical               0                     │
│ High                   2                     │
│ Medium                 4                     │
│ Low                    2                     │
└──────────────────────────────────────────────┘
```

They display:
- **Assets**: Authentication Service, Public API...
- **Threats**: Potential Authentication Abuse...
- **Evidence**: Observed POST /api/login...
- **Attack path**: Internet ↓ Public Login Endpoint ↓ Authentication Service ↓ User Account
- **Recommendations**: Implement rate limiting...

---

## 20. What exactly Member 2 needs to build

Member 2 owns `app/discovery/` and `backend/`.

A. URL handling
B. Basic HTTP discovery
C. Page discovery
D. Link extraction
E. Form discovery
F. Endpoint indicators
G. Technology fingerprinting
H. Security indicators
I. Produce DiscoveryResult

---

## 21. Member 2 also owns backend orchestration

This is separate from discovery. Their backend eventually coordinates:
Frontend ↓ Backend ↓ Discovery ↓ Your Security Analysis ↓ Backend ↓ Frontend

---

## 23. What exactly Member 3 needs to build

Member 3 owns `frontend/`. Their job is the user-facing application.

A. URL input screen
B. Analysis state
C. Results dashboard
D. Asset view
E. Threat view
F. Risk visualization
G. Attack-path visualization
H. Recommendations
I. Reports

---

## 25. The shared boundary between all three

**Boundary #1**
Member 2 ↓ DiscoveryResult ↓ YOU
Member 2 must give you structured observations.

**Boundary #2**
YOU ↓ SecurityAnalysis ↓ Member 3
You must give Member 3 structured security analysis.

---

## 27. The three people’s code should look roughly like this

```text
Automated-Threat-Modeling/
│
├── app/
│   ├── models/                  ← SHARED
│   │   └── schemas.py
│   ├── discovery/               ← MEMBER 2
│   ├── security/                ← YOU
│   ├── rag/                     ← YOU
│   └── ai/                      ← YOU
├── knowledge_base/              ← YOU
├── backend/                    ← MEMBER 2
├── frontend/                   ← MEMBER 3
└── tests/
```

---

## 31. GitHub workflow

```text
                    GitHub
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       YOUR PC     Member 2 PC  Member 3 PC
          │           │           │
    your branch   their branch  their branch
```
Each person’s AI works on their own local branch.

---

## 34. The final ownership table

| Component | You | Member 2 | Member 3 |
| :--- | :--- | :--- | :--- |
| Project architecture | 🟢 | 🟢 | 🟢 |
| Shared schemas | 🟢 | Review | Review |
| URL input UI | ❌ | ❌ | 🟢 |
| Web discovery | ❌ | 🟢 | ❌ |
| Backend API | ❌ | 🟢 | ❌ |
| Asset intelligence | 🟢 | ❌ | ❌ |
| Threat modeling | 🟢 | ❌ | ❌ |
| Security KB | 🟢 | ❌ | ❌ |
| RAG | 🟢 | ❌ | ❌ |
| LLM reasoning | 🟢 | ❌ | ❌ |
| Deterministic risk engine | 🟢 | ❌ | ❌ |
| Attack-path intelligence | 🟢 | ❌ | ❌ |
| Attack-path visualization | ❌ | ❌ | 🟢 |
| Risk dashboard | ❌ | ❌ | 🟢 |
| Threat UI | ❌ | ❌ | 🟢 |
| Recommendations UI | ❌ | ❌ | 🟢 |
| Report UI | ❌ | Backend support | 🟢 |
| Final integration | 🟢 | 🟢 | 🟢 |

---

## 35. The simplest explanation you can give your teammates

**Member 2:** “Your job is to build the eyes and backend. Given an authorized URL, safely discover observable information and turn it into the shared DiscoveryResult. Don’t decide whether anything is a threat.”

**Me:** “My job is the cybersecurity and AI brain. I take DiscoveryResult, identify security assets, model potential threats, retrieve OWASP/CWE/MITRE knowledge using RAG, use an LLM for reasoning, calculate risk deterministically, construct potential attack paths, and produce SecurityAnalysis.”

**Member 3:** “Your job is the interface. Take the SecurityAnalysis and make a dashboard showing assets, threats, risk, evidence, attack paths, and recommendations. Don’t implement the cybersecurity reasoning yourself.”

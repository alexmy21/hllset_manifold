## 2. The HLLSet Framework

We introduce **HLLSet** (HyperLogLog Set) – a probabilistic data structure that **behaves like a set under all standard operations yet contains no elements**. Instead of storing its members, an HLLSet is a fixed‑size fingerprint: a vector of \(m\) integers, each of \(b\) bits. Every token contributes **exactly one bit** at a position derived deterministically from its hash. Because the fingerprint does **not** identify which token inscribed that bit, many tokens may occupy the same position – this is not a defect, but the source of the structure’s capacity to generalise and to represent arbitrarily large sets in bounded space.

We call this an **anti‑set**: a dual object that stands for a set without being one. Its mathematical foundation is the **idempotence** of the inscription procedure: the same token always maps to the same bit, and adding it twice does not change the fingerprint. This idempotence allows us to **forget** the tokens entirely and work purely in the algebra of fingerprints.

---

### 2.1 HLLSet Fingerprint and Inscription

**Definition 2.1 (HLLSet).**  
An HLLSet is determined by two global parameters:

- \(m\) – number of **registers** (typically a power of two);  
- \(b\) – width of each register in bits (e.g., 16, 32, 64).

Its state is a vector of \(m\) \(b\)-bit integers:

\[
\mathbf{R} = (R_1, R_2, \ldots, R_m), \qquad R_i \in \{0,1\}^b.
\]

Equivalently, we may view \(\mathbf{R}\) as an \(m \times b\) bit matrix.  
The **empty anti‑set** \(\varnothing\) has \(\mathbf{R} = \mathbf{0}\).

---

**Inscription of a token.**  
Let \(\mathcal{T}\) be the universe of tokens (byte strings). For each token \(t \in \mathcal{T}\) we compute a hash \(h = \mathsf{hash}(t)\) of length at least \(\log_2 m + b\) bits.

1. **Register index:** \(i = h \bmod m\).  
2. **Bit position:** Take the remaining bits \(q = h \gg \lceil\log_2 m\rceil\) and compute \(z = \nu(q)\) – the number of trailing zeros in the binary representation of \(q\), capped at \(b-1\) (i.e., \(z = \min(\nu(q), b-1)\)).  

The **singleton anti‑set** \(\{\!|t|\!\}\) is defined as the HLLSet having a single 1‑bit at register \(i\), position \(z\), and zeros elsewhere:

\[
(\{\!|t|\!\})_i = 1 \ll z,\qquad (\{\!|t|\!\})_{j \neq i} = 0.
\]

**Key property: one token, one bit.** Every token occupies **exactly one bit** in the fingerprint. Different tokens may, and often do, map to the same \((i,z)\) pair – this is a **collision**. Collisions are intrinsic and desirable; they create the controlled ambiguity that enables generalisation and the directed similarity measure (BSS, §2.4).

---

**Building an HLLSet from a sequence of tokens.**  
HLLSets are **immutable**: once constructed, a fingerprint never changes. To obtain an HLLSet that contains additional tokens, we perform **union** with the singleton anti‑sets of those tokens. Union is defined as bitwise OR on the register vectors:

\[
\mathbf{R}_{A \cup B} = \mathbf{R}_A \vee \mathbf{R}_B.
\]

Thus, given a sequence of tokens \(t_1 t_2 \ldots t_k\), we define

\[
\{\!|t_1, t_2, \ldots, t_k|\!\} \;:=\; \{\!|t_1|\!\} \cup \{\!|t_2|\!\} \cup \cdots \cup \{\!|t_k|\!\}.
\]

Because OR is associative, commutative, and idempotent, the resulting HLLSet depends only on the **set** of tokens, not on their order or multiplicity.

---

**Tokenization functor.**  
Let \(\mathsf{Seq}(\mathcal{T})\) be the free monoid over tokens (finite sequences, concatenation as product).  
Define \(\phi : \mathsf{Seq}(\mathcal{T}) \to \mathbf{HLLSet}\) by:

\[
\phi(\varepsilon) = \varnothing,\qquad
\phi(t_1 t_2 \ldots t_k) = \{\!|t_1|\!\} \cup \{\!|t_2|\!\} \cup \cdots \cup \{\!|t_k|\!\}.
\]

**Theorem 2.2 (Functoriality).**  
\(\phi\) is a monoid homomorphism:

\[
\phi(s \cdot t) = \phi(s) \cup \phi(t), \qquad \phi(\varepsilon) = \varnothing.
\]

*Proof.* Immediate from the definition of union as OR and the associativity/commutativity of OR. ∎

This functor is the formal bridge between the concrete world of tokens and the abstract, token‑agnostic world of HLLSet fingerprints.

---

### 2.2 Why “Anti‑Set”? The Inversion of Definition

A classical set is defined **by its elements**; its identity is the collection of its members.  
An HLLSet is defined **by its fingerprint**; the fingerprint does **not** directly contain its elements, yet under the operations of union, intersection, and difference it behaves **exactly as if it were the set**. This inversion – from “set of elements” to “element of a space of fingerprints” – is the essence of the anti‑set.

This inversion is possible only because the inscription procedure is **idempotent** and **deterministic**:

- **Idempotence:** Adding the same token twice yields the same fingerprint as adding it once.  
- **Determinism:** The same token always maps to the same bit.

These properties allow us to **forget the tokens** and work entirely within the algebra of fingerprints. The image of \(\phi\) is a **Karoubi completion** of the idempotent hash functions (see §6.1); the anti‑set is therefore not an implementation trick but a mathematically well‑founded dual object.

**From this point onward, we rarely mention tokens.** All constructions (similarity, lattices, dynamics in the AM pipeline) are expressed purely in terms of HLLSet fingerprints and their bitwise operations. This token‑agnosticism is the hallmark of the framework.

---

### 2.3 Set Algebra on HLLSets

Because HLLSet fingerprints are vectors of bit‑vectors, all classical set operations are implemented **deterministically** via bitwise Boolean logic on the registers:

- **Union:** \(\mathbf{R}_{A \cup B} = \mathbf{R}_A \vee \mathbf{R}_B\).  
- **Intersection:** \(\mathbf{R}_{A \cap B} = \mathbf{R}_A \wedge \mathbf{R}_B\).  
- **Difference:** \(\mathbf{R}_{A \setminus B} = \mathbf{R}_A \wedge \neg(\mathbf{R}_B \wedge \mathbf{R}_A)\).

These operations are associative, commutative, and idempotent; they satisfy all the usual identities of set algebra **when interpreted on the underlying token sets**, up to the bounded error introduced by hash collisions. The algebra is **closed**: the result of any operation is again an HLLSet with the same parameters \((m,b)\).

**Cardinality estimation.**  
The true cardinality \(|A|\) – the number of distinct tokens inscribed – is not directly accessible, but can be estimated from the fingerprint. The classic HyperLogLog estimator [1] uses only the maximum trailing‑zero count per register; with the full bit vector \(\mathbf{R}_A\) more accurate estimators exist (e.g., [2]). We denote the estimator by \(\widehat{|A|}\); it is unbiased and its relative error is \(O(1/\sqrt{m})\).

---

### 2.4 Bell State Similarity (BSS)

A central innovation of HLLSet theory is a **directed** similarity measure that captures the degree of containment between two anti‑sets.

**Definition 2.3 (Bell State Similarity).**  
For HLLSets \(A,B\) define  

\[
\mathrm{BSS}_\tau(A \to B) = \frac{\widehat{|A \cap B|}}{\widehat{|B|}}, \qquad
\mathrm{BSS}_\rho(A \to B) = \frac{\widehat{|A \setminus B|}}{\widehat{|B|}}.
\]

If \(\widehat{|B|} = 0\) we set \(\mathrm{BSS}_\tau = 0\) and \(\mathrm{BSS}_\rho = 1\).

\(\mathrm{BSS}_\tau\) measures the *inclusion* of \(A\) in \(B\); \(\mathrm{BSS}_\rho\) measures the *exclusive part* of \(A\) relative to \(B\). Both are estimated quantities, inheriting the error bounds of the cardinality estimator.

**Definition 2.4 (Morphism).**  
Given two HLLSets \(A,B\) with associated thresholds \((\tau_A,\rho_A)\) and \((\tau_B,\rho_B)\), a **morphism** \(f: A \to B\) exists iff  

\[
\mathrm{BSS}_\tau(A \to B) \ge \tau_A \quad\text{and}\quad \mathrm{BSS}_\rho(A \to B) \le \rho_B.
\]

The identity \(1_A : A \to A\) always exists because \(\mathrm{BSS}_\tau(A\to A)=1\) and \(\mathrm{BSS}_\rho(A\to A)=0\).

Morphisms do **not** compose automatically; composition depends on the transitivity of the threshold conditions. Hence the collection of HLLSets with morphisms forms a **directed graph** rather than a category. This graph is the raw material from which higher‑order structure emerges.

---

### 2.5 The W Lattice

The set of all HLLSet fingerprints (for fixed \(m,b\)) is partially ordered by **bitwise inclusion**:

\[
A \le B \;\Longleftrightarrow\; \mathbf{R}_A \wedge \neg \mathbf{R}_B = 0.
\]

This is a distributive lattice (meet = \(\wedge\), join = \(\vee\)) with bottom \(\varnothing\) and top the all‑ones matrix (which corresponds to the “universe” of all possible bits). We call this the **W lattice**. Its elements are fingerprints; its order reflects true set inclusion when collisions are ignored. BSS provides a real‑valued directed similarity measure on this lattice.

---

### 2.6 Summary

The HLLSet framework provides a **static, algebraic** representation of sets. An HLLSet is an anti‑set: it behaves like a set under union, intersection, and difference, yet it contains no elements. Its identity is its fingerprint, a fixed‑size bit vector. The framework is **token‑agnostic**: once the tokenization functor \(\phi\) is applied, all further operations and constructions (BSS, W lattice) are expressed purely in terms of fingerprints.  

Because HLLSets are immutable, there are **no primitive update operations** – only the static algebra. Any dynamics must come from an external environment that **generates new HLLSets** from streams of tokens. The next section presents one such environment – the Adjacency Matrix pipeline – where token pairs are added and removed in a balanced fashion, giving rise to a discrete conservation law that steers the evolution of the derived W lattice.

## 3. Instantiation I: The Adjacency Matrix Pipeline and Noether Steering

The HLLSet framework provides a static, token‑agnostic algebra of anti‑sets. To build a dynamical AI system we must **embed** this algebra in an environment that generates tokens and consumes the resulting fingerprints. This section presents a concrete instantiation – the **Adjacency Matrix (AM) pipeline** – in which:

- The environment emits a stream of **ordered token pairs** \((i,j)\) (e.g., a token and its successor in a text).  
- A co‑occurrence matrix \(C\) records the frequency of each pair.  
- From \(C\) we construct, for each token \(i\), an HLLSet \(S_i^+\) representing the **set of tokens that have followed \(i\)**.  
- These HLLSets are organised into a **W lattice** using Bell State Similarity.  
- The W lattice becomes the system’s **active cognitive model**: it supports similarity queries, inference, and generation.  
- Crucially, the environment may also **remove** previously observed pairs. When additions and deletions are balanced, the system exhibits a **conservation law** – a discrete analogue of Noether’s theorem – that acts as a **steering principle** for the entire pipeline.

This instantiation is not the framework itself; it is one compelling demonstration of how the HLLSet framework can be coupled with a dynamical environment to produce an auditable, structurally aware AI component.

---

### 3.1 Environment and the Co‑occurrence Matrix

Let \(\mathcal{T}\) be a finite set of token types (or token hashes). The **environment** emits two types of events:

- **Addition** of a token pair \((i,j)\) – indicating that token \(j\) was observed immediately after token \(i\).  
- **Deletion** of a token pair \((i,j)\) – indicating that a previously recorded co‑occurrence is removed (e.g., because a document is deleted, or because the system implements a sliding window).

We maintain a **co‑occurrence matrix** \(C \in \mathbb{N}^{|\mathcal{T}| \times |\mathcal{T}|}\) (or, more practically, a sparse representation) that is updated as follows:

\[
C_{ij} \;\leftarrow\; C_{ij} + 1 \quad \text{on addition of }(i,j),
\]
\[
C_{ij} \;\leftarrow\; C_{ij} - 1 \quad \text{on deletion of }(i,j).
\]

We assume that deletions never cause a count to become negative; that is, the environment is **well‑behaved**. The matrix \(C\) is the **raw memory** of the system – it records the current multiset of observed co‑occurrences.

---

### 3.2 Building HLLSets from the Co‑occurrence Matrix

For each token \(i \in \mathcal{T}\), we define its **successor HLLSet** \(S_i^+\) as the anti‑set that represents the set of tokens that have ever followed \(i\) (with multiplicities ignored, because HLLSets are idempotent).

**Construction.**  
Start with the empty HLLSet \(\varnothing\). For each column \(i\) of \(C\), scan all tokens \(j\) such that \(C_{ij} > 0\). For each such \(j\), add the singleton \(\{\!|j|\!\}\) to \(S_i^+\):

\[
S_i^+ = \bigcup_{j : C_{ij} > 0} \{\!|j|\!\}.
\]

Because union is commutative, associative, and idempotent, the resulting \(S_i^+\) depends only on the **set** of tokens that have followed \(i\), not on their frequencies or order of insertion. Moreover, the construction is **idempotent** in the following sense:

**Proposition 3.1 (Idempotence of the AM→HLLSet mapping).**  
If a token pair \((i,j)\) is added twice (or deleted and then re‑added) without any intervening change to the presence of other tokens in the column, the resulting \(S_i^+\) is the same as if it had been added once.

*Proof.* Follows directly from the idempotence of the union operation: \(\{\!|j|\!\} \cup \{\!|j|\!\} = \{\!|j|\!\}\). ∎

Thus the mapping from a column of \(C\) to an HLLSet is a **perfect hash‑based summarisation** that discards multiplicities but retains membership information up to collision ambiguity.

---

### 3.3 The W Lattice from Successor HLLSets

Let \(\mathcal{S}^+ = \{ S_i^+ \mid i \in \mathcal{T} \}\) be the collection of successor HLLSets. We now organise these anti‑sets into a **W lattice** using the Bell State Similarity defined in §2.4.

For every ordered pair \((i,j)\) we compute the estimated inclusion and exclusion ratios:

\[
\mathrm{BSS}_\tau(S_i^+ \to S_j^+) = \frac{\widehat{|S_i^+ \cap S_j^+|}}{\widehat{|S_j^+|}}, \qquad
\mathrm{BSS}_\rho(S_i^+ \to S_j^+) = \frac{\widehat{|S_i^+ \setminus S_j^+|}}{\widehat{|S_j^+|}}.
\]

Choose global thresholds \(\tau, \rho\) (or, more generally, per‑node thresholds). A **directed edge** \(S_i^+ \to S_j^+\) exists in the **W graph** iff

\[
\mathrm{BSS}_\tau(S_i^+ \to S_j^+) \ge \tau \quad\text{and}\quad \mathrm{BSS}_\rho(S_i^+ \to S_j^+) \le \rho.
\]

This graph inherits the partial order of the W lattice (bitwise inclusion) and adds a graded, directed similarity relation. We refer to it simply as the **W lattice** of the system.

**Why this is the active cognitive model.**  
The W lattice encodes **containment relationships** between the successor profiles of different tokens. A high \(\mathrm{BSS}_\tau(S_i^+ \to S_j^+)\) means that most of the tokens that follow \(i\) also follow \(j\) – i.e., \(j\)’s distribution “covers” \(i\)’s. This is a form of **semantic entailment** or **generality hierarchy**. The W lattice is therefore a **structural summary** of the sequential patterns in the environment, and it can be queried without any further reference to the raw co‑occurrence matrix.

---

### 3.4 The Symmetry of Balanced Update

The environment may add and remove token pairs. Consider the total number of recorded co‑occurrences:

\[
\mathcal{C} = \sum_{i,j} C_{ij}.
\]

**Theorem 3.2 (Conservation of total co‑occurrence count).**  
Let the environment be **closed** in the sense that every addition of a pair \((i,j)\) is eventually matched by a deletion of the same pair, and vice versa, with no net change in the multiset of pairs. Then \(\mathcal{C}\) is constant.

*Proof.* Each addition increments \(\mathcal{C}\) by exactly 1; each deletion decrements \(\mathcal{C}\) by exactly 1. If the multiset of additions equals the multiset of deletions, the total change is zero. ∎

This is a trivial arithmetic fact, yet it carries profound implications when viewed through the lens of Noether’s theorem. The update dynamics are invariant under the simultaneous, equal addition and removal of the same token pair – a discrete \(\mathbb{Z}_2\) symmetry. The conserved quantity is the total co‑occurrence count \(\mathcal{C}\).

**Definition 3.3 (Noether steering law).**  
We call the invariance of \(\mathcal{C}\) under balanced updates the **Noether steering law** for the AM pipeline. It serves as:

- a **monitor**: any sustained drift in \(\mathcal{C}\) signals an imbalance between additions and deletions, which may indicate a bug, an intentional growth/decay phase, or an external flux of information;
- a **design principle**: long‑term stable systems must balance their update streams.

---

### 3.5 Propagation to the W Lattice

The W lattice is derived deterministically from the successor HLLSets \(S_i^+\), which are themselves derived from the columns of \(C\). Because the mapping from \(C\) to \(\{S_i^+\}\) is idempotent and purely set‑based, it is **insensitive to frequency multiplicities**. A deletion of a pair \((i,j)\) only affects \(S_i^+\) if it is the **last occurrence** of that pair; similarly, an addition only affects \(S_i^+\) if the pair was previously absent.

Thus the W lattice changes only when a token pair **appears for the first time** or **disappears entirely**. In a closed environment where the multiset of pairs is constant, the set of present pairs is also constant, and therefore the W lattice is **invariant**. This gives a stronger conservation result:

**Corollary 3.4 (W‑lattice conservation in a closed environment).**  
If the multiset of token pairs is constant (additions and deletions balance exactly, including multiplicities), then the successor HLLSets \(\{S_i^+\}\) and the resulting W lattice are unchanged.

*Proof.* The successor HLLSets depend only on which tokens have non‑zero counts in each column. If the multiset of pairs is constant, the set of tokens with non‑zero counts in each column is also constant. Hence the \(S_i^+\) are unchanged, and so is the W lattice derived from them. ∎

In practice, the environment may not be perfectly closed, but the **drift** in \(\mathcal{C}\) provides an immediate bound on the rate of structural change in the W lattice. Developing tight bounds is a subject of ongoing work.

---

### 3.6 Closing the Loop: From W Lattice to Action

The W lattice is not a passive data structure; it is the system’s **internal representation** of the environment’s sequential structure. To act on the environment – for example, to generate the next token in a sequence – the system queries the W lattice.

**A typical generation step:**

1. **Current context** is represented as an HLLSet \(C_{\text{ctx}}\) (e.g., the successor set of the current token, or a mixture of recent tokens).
2. Find the token \(i\) whose successor HLLSet \(S_i^+\) is **most similar** to \(C_{\text{ctx}}\) under the BSS measure (or a symmetric variant).
3. From the HLLSet \(S_i^+\), **sample** a token that is contained in it. (This requires a reverse mapping from bits to tokens; we have developed a practical solution based on maintaining a mapping from each bit position to the set of tokens that have inscribed it, but this detail is outside the scope of the present paper.)
4. Emit the sampled token, which becomes part of the environment and may be recorded as a new token pair.

This closes the cycle:  
\[
\text{Environment} \;\to\; \text{AM} \;\to\; \text{W lattice} \;\to\; \text{Action} \;\to\; \text{Environment}.
\]

The Noether steering law ensures that, if the environment is closed, the system’s internal model (the W lattice) remains stable. If the environment is growing or shrinking, the drift in \(\mathcal{C}\) provides a direct measure of the net information flux, which can be used to adapt the system’s behaviour.

---

### 3.7 Summary

The AM pipeline is a concrete instantiation of the HLLSet framework in a dynamical setting. It demonstrates:

- How raw sequential data (token pairs) can be **compressed** into HLLSet fingerprints.
- How these fingerprints are organised into a **W lattice** that captures containment relationships.
- How the **symmetry** of balanced addition/removal gives rise to a **conservation law** – the Noether steering law – which provides a principled monitor and design constraint.
- How the W lattice can be used to **generate actions**, closing the perception–action loop.

This instantiation is not the only one; the next section briefly outlines other domains where the HLLSet framework provides similar benefits. The Noether steering law, however, is specific to environments that support both addition and deletion of tokens. In purely additive settings (e.g., static corpora), the conservation law is trivial (no deletions, hence \(\mathcal{C}\) only increases) and the steering principle reduces to a monotonic growth monitor.
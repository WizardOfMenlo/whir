# Zero-Knowledge WHIR with poly-logarithmic overhead

This document describes a space and time efficient version of Zero-Knowledge WHIR. The rest of this document assumes familiarity with the [previous version of Zero-Knowledge WHIR](https://www.notion.so/Zero-Knowledge-for-WHIR-1b18614bdf8c80d4acffd09c7c55da1c?pvs=21) (called zkWHIR-1.0) and uses notation from that document. While zkWHIR-1.0 is $4$-times slower than non Zero-Knowledge WHIR, the new scheme only adds poly-logarithmic space and time overhead to non-ZK WHIR.

## Optimization Overview

Conceptually, the new scheme is still based on zkWHIR-1.0, but with the following two optimizations:

1. The first optimization eliminates the need for an extra variable $y$ that was necessary for *soundly* adding a masking polynomial $\mathsf{msk}(\vec{x})$ to $f(\vec{x})$.

The key observation is that FRI's $\mathsf{fold}_k(f, \vec{r})$ operation is *linear*. Therefore, one can mask $f(\vec{x})$ by computing $[[f + \textsf{msk}]]$ oracle directly and *additionally* committing to $\textsf{msk}(\vec{y})$ of small degree (roughly size of the query complexity of WHIR). However, the challenge of validating claims involving $[[f]]$ *without* opening $\textsf{msk}$ at oracle query points still remains.

To address this, instead of committing to $\textsf{msk}$ directly, the prover commits to a new polynomial $M(\vec{y}, {\color {Magenta}t}) := g(\vec{y}) + {\color {Magenta} t}\cdot\textsf{msk}(\vec{y}) \in \mathbb{F}_q^{\preceq 1}[\vec{y}, {\color {Magenta}t}]$ for some random $g(\vec{y})$ of small degree. During first round of Sumcheck, when the Prover sends it first folded oracle $\textsf{fold}_k(\rho\cdot f + g, \vec{r})$, the verifier can validate claims about $[[f + \textsf{msk}]]$ using the following identity:
    
    $$
    {\color{Magenta}\rho}\cdot \underbrace{\textsf{fold}_k(f + \textsf{msk},\vec{r})}_{\text{computed from } [[f+\textsf{msk}]]} + \underbrace{\textsf{fold}_k(g-{\color{Magenta}\rho}\cdot\textsf{msk}, \vec{r})}_{\text{direct computation}\,M(\vec{y}, -\rho)} =  \underbrace{\textsf{fold}_k(\rho\cdot f + g, \vec{r})}_{\text{first sumcheck oracle}} 
    $$
    
    where the value of $M(\vec{y}, -\rho)$ is computed (with proof) by the prover and locally folded by the verifier.
    
    Even though $M(\vec{y}, t)$ has one extra variable than $\textsf{msk}$, since the degree of $\textsf{msk}$ is much smaller than that of $f$, the size of $[[M]]$ is substantially smaller than the size of $[[\widehat{f}]]$ in zkWHIR-1.0. 
    
    **NOTE**: This scheme is only sound if degree of $f$ is greater than the degree of $\textsf{msk}$, which is roughly equal to the query complexity of WHIR. If the degree of $f$ is as small as the query-complexity, then one should use zkWHIR-1.0 which is equally efficient in that parameter range.
    
2. The second optimization eliminates the need to commit to a large-degree sparse blinding polynomial. The key observation is that FRI's *per-round* query complexity is constant (i.e., independent of oracle size). Therefore, for large oracles, it's more efficient to commit to $\mu + 1$ random polynomials—each roughly the size of the per-round query complexity—and construct the blinding polynomial on the fly rather than commit to it directly.

Let $g_0(X),g_1(X),\cdots, g_{\mu}(X)$ (written in univariate form) be $\mu+1$ random polynomials, where the degree of each $g_i(X)$ roughly equals the query complexity of WHIR. Given $g_i$, the zkWHIR-1.0 Sumcheck blinding polynomial can be constructed as:
    
    $$
    g(X) = g_0(X) + \sum_{i=1}^{\mu} \beta^{i}\cdot X^{2^{i-1}}\cdot g_i(X) 
    
    $$
    
    where $\beta$ is verifier-selected randomness. Furthermore, given polynomial commitment to $g_i$, the verifier can locally evaluate $g(\vec{x})$ by opening $g_i$s at the desired points and validating its proof.
    
    To summarize, instead of computing the oracle $[[g]]$ and its Merkle opening proofs at FRI query points, the verifier opens $g_i(X)$s at those points and locally constructs $[[g]]$. This representation:
    
    1. Replaces a single large NTT computation with evaluations of multiple small-degree polynomials at FRI query points, and
    2. Replaces the Merkle opening proofs with a single *batched WHIR opening proof* of evaluations of $g_i$s at query points. Note that since each $g_i$ is evaluated at the same point, they can be batched very efficiently. Furthermore, since $g(X)$ is evaluated at query-complexity number of points, those openings could also be batched together (using random linear combination), resulting in a single giant batched WHIR opening proof.
    
    To establish Zero-Knowledge, note that each $g_i$ has a large enough degree that its evaluations are perfectly indistinguishable from random. To simulate the distribution of Sumcheck rounds, if the constant term of $g_i$s are non-zero, the multilinear representation of $g$ will have random non-zero coefficient for each indeterminate $x_i$, i.e, $g(x_1, \cdots, x_\mu)$ is of the form
    
    $$
    a_0 + a_1x_1 + a_2x_2 + \cdots + a_\mu x_\mu + \cdots
    $$
    
    where *each $a_i$ is uniformly random* (and non-zero with high probability). This ensures that Sumcheck round polynomials for the claim $\rho\cdot f + g$ are blinded using a uniformly random linear polynomial, allowing the ZK Simulator to simulate this interaction without access to witness $f$.
    

For the rest of the document, the reader is assumed to be familiar with [zkWHIR-1.0](https://www.notion.so/Zero-Knowledge-for-WHIR-1b18614bdf8c80d4acffd09c7c55da1c?pvs=21).

### Notation

Lower case variables (e.g., $x_1, \cdots x_t$) denote the indeterminates of a multivariate polynomial, while upper case variables (e.g., $X$ or $Y$) denote the indeterminates of a univariate polynomial. Abusing terminology, the degree of multilinear $h(\vec{x})$ polynomial refers to the degree of the univariate polynomial $h(X^{2^0}, X^{2^1}, \cdots, X^{2^{t-1}})$.

<aside>
👉

Recall that given a $t$-variate *multilinear* polynomial $\widehat{h}(x_1,\cdots,x_t)$ , there exits a change of basis (a bijective map)  $\mathsf{U}_{t,X}(\widehat{h})$  that maps $\widehat{h}(x_1,…x_t)$ to its univariate representation $h(X)$ as follows:

$$
\begin{aligned}
\mathsf{U}_{t,X} &: \mathbb{F}_q^{\preceq 1}[x_1,\cdots,x_t] \longrightarrow  \mathbb{F}_q^{\preceq 2^{t}-1}[X] \\ 
\mathsf{U}_{t,X}(h) &:= h(X^{2^0},X^{2^1}, \cdots, X^{2^{t-1}})\end{aligned}
$$

The inverse map $\mathsf{U}_{t,X}^{-1}$ is given by

$$
\begin{aligned}
\mathsf{U}_{t,X}^{-1} &: \mathbb{F}_q^{\preceq 2^{t} -1}[X] \longrightarrow F_q^{\preceq 1}[x_1,\cdots, x_t] \\
\mathsf{U}_{t,X}^{-1}\left (\sum_{i=0}^{2^{t}-1} a_i X^i\right) &:=  
\sum_{i=0}^{2^{t}-1} a_i \left (\prod_{j \in [t]} x_j^{\textsf{bits}(i)[j]}\right)
\end{aligned}
$$

where $\mathsf{bits}(i)$ represents the $t$-bit binary representation of $i$ and $\mathsf{bits}(i)[j]$ represents the $j$-the component of that binary vector.

**NOTE**: $\mathsf{U}_{t,X}$ (and its inverse) is a bijective map from one bounded-degree polynomial to another, where the bounded degree polynomial is treated as a vector space. However, the *polynomial functions* $h(X)$ and $\widehat{h}(x_1,\cdots, x_t)$ are very different functions (their domains are dramatically different)! This mapping is also not the multilinear extension of $h(X)$!

</aside>

## zkWHIR 2.0 Protocol

zkWHIR 2.0 makes use of two different instances of WHIR as a subprotocol. The first instance runs over the $\mu$-variate witness polynomial $f(x_1, \cdots, x_\mu)$. The second instance runs over randomly generated *masking* and *blinding* polynomials. Let$y_1,\cdots y_\ell$$\ell \ll \mu$ denote the number of variables needed to represent the masking and blinding polynomials. Let $k = 2^s$ be the folding factor for first round of WHIR where $s > 0$.

The value of $\ell$ depends upon the query complexity of FRI oracles and details about its computation is described below:

<aside>
🛠

### Query Complexity Computation

Let $\delta_1$ denote the decoding radius of first WHIR instance and let $\mathsf{q}(\delta_1)$ be the corresponding query complexity. Let $\delta_2$ be the decoding radius of second WHIR instance and $\mathsf{q}(\delta_2)$ the corresponding query complexity. Since proximity gaps up to list decoding are well established, it’s safe to use Johnson radius as $\delta_i$s.

Since the query complexity of FRI is independent of the length of the codeword, when the degree of folded polynomial reaches a certain size, it’s more efficient to send the entire folded polynomial in clear than the corresponding codeword. Let $T_1 \ge \mathsf{q}(\delta_1)$ denote the size of the polynomial that’s sent in raw during the last round of the first instance of WHIR and let $T_2 \ge \mathsf{q}(\delta_2)$ denote this number for the second instance of WHIR.

Let 

$$
\textsf{q}_{\textsf{ub}} = k\cdot \mathsf{q}(\delta_1) + \mathsf{q}(\delta_2) + T_1 + T_2 + 4\mu
$$

and $\ell$ be an integer such that 

$$
2^\ell > \textsf{q}_{\textsf{ub}}
$$

The $\mathsf{q}_{\textsf{ub}}$ represents a highly *conservative* upper bound on the number of points where any polynomial needs to be evaluated. This value guarantees that evaluated polynomials constitute an under constrained system of linear equations. This is essential for ZK Simulator.

</aside>

Other parameters needed for the protocol are listed below with their typical values:

<aside>
🛠

### WHIR Parameters

A Reed-Solomon code defined over an evaluation domain $\Omega$ of size $N$ has (relative) code rate $\rho := \frac{d + 1}{N} = \frac{d}{N} + o(1)$. The minimum distance between any two codewords is $\delta_{\mathsf{min}} := \frac{N - d}{N} = 1 - \frac{d}{N} = 1 - \rho - o(1)$. 

The following table describes the list of relevant parameters with their typical values for Reed Solomon (RS) Code.

**NOTE**: The query complexity is independent of the length $N$ of codeword.

| **Parameter Name** | **Symbol** | **Typical Values** |
| --- | --- | --- |
| Security Parameter | $\lambda$ | 100 bits (i.e., probability of forging $2^{-100})$ |
| RS Code Rate | $\rho$ $(0 < \rho \leq \frac{1}{2})$ | $\frac{1}{2}$ |
| Decoding Radius | $\delta$ $(0 < \delta < 1-\rho)$ | Up to list-decoding radius $\delta_{\mathsf{ld}} := 1-\sqrt{\rho}$ |
| Query Complexity | $\mathsf{q}(\delta) := \left\lceil \frac{\lambda}{\log_2(\frac{1}{1-\delta})} \right \rceil$ | $\mathsf{q}_{\textsf{ld}}$ |
| Query Complexity up to List Decoding | $\begin{aligned}\mathsf{q}_{\mathsf{ld}} &:= \mathsf{q}(1-\sqrt{\rho}) \\ &=  \left\lceil \frac{2\lambda}{\log_2(1/\rho)}\right\rceil \end{aligned}$ | 200 |
| Query Complexity up to Unique Decoding | $\begin{aligned}
\mathsf{q}_{\mathsf{ud}} &:= \mathsf{q}\left (\left \lfloor\frac{1-\rho}{2}\right \rfloor \right) \\ &= \left\lceil\frac{\lambda}{1 - \log_2(1+\rho)}\right\rceil \end{aligned}$  | 241 |
| Folding Factor | $k = 2^s\; (s > 0)$ | $k = 4$; $(s = 2)$ |
| Last fold size | $T1, T2$ | $\textsf{q}_{\textsf{ld}}$ |
| Query upper bound | $\textsf{q}_{\textsf{ub}}$ | 1420 |
| Small poly variable count | $\ell$ | 11 |
</aside>

## Randomness Sampling/Preprocessing:

1. The (honest) Prover ($\mathsf{P}$) samples following random polynomials:
    1. An $\ell$-variate multilinear ***masking polynomial***
        
        $$
        \mathsf{msk}(y_1,\cdots,y_\ell) \xleftarrow{\$} \mathbb{F}_q^{\preceq 1}[y_1,\cdots, y_{\ell}]
        $$
        
    2. $\mu + 1$ additional univariate polynomials of degree of $2^{\ell}-1$ as
        
        $$
        \begin{aligned}  g_0(X) &\xleftarrow{\$} \mathbb{F}_q^{\prec 2^\ell}[X] \\ & \quad\vdots \\ g_\mu(X) &\xleftarrow{\$} \mathbb{F}_q^{\prec 2^\ell}[X] \\ \end{aligned}
        $$
        
        and let $\widehat{g}_i(\vec{y})$ be its corresponding multivariate representation, i.e.,   
        
        $$
        \begin{aligned} \widehat{g}_0, \cdots, \widehat{g}_{\mu} &\in \mathbb{F}_q^{\preceq 1}[y_1,\cdots,y_\ell] \\ \widehat{g}_i(y_1,\cdots,y_\ell) &= \mathsf{U}_{\ell,X}^{-1}(g_i) \end{aligned}
        $$
        
         where $\mathsf{U}_{\ell, X}(\cdot)$ is defined in the [notation section](https://www.notion.so/Zero-Knowledge-WHIR-with-poly-logarithmic-overhead-2e28614bdf8c80be8c95edaa2fd93f9f?pvs=21).  
        
    3. $\mathsf{P}$ computes the following commitment polynomial
        
        $$
        M(\vec{y};\; t) := \widehat{g}_0(\vec{y}) + {\color{Magenta} t}\cdot \textsf{msk}(\vec{y}) \in \mathbb{F}_q^{\preceq 1}[y_1,\cdots,y_\ell,\; {\color{Magenta} t}]
        $$
        
    4. $\mathsf{P}$ computes $\mu+1$ oracles using $\ell + 1$ variate  polynomials. 
    
    $$
    \begin{aligned}
    [[M]] &:= \textsf{commit}(M) \\ [[\widehat{g}_1]] &:= \textsf{commit}(\widehat{g}_1) \\ &\;\vdots \\ [[\widehat{g}_\mu]] &:=\textsf{commit}(g_\mu)
    \end{aligned}
    $$
    
    Even though $g_i$’s are $\ell$-variate, they should be treated as $\ell+1$ variate polynomials where the coefficient corresponding to variable $t$ is set to zero. Notice that there’s no independent commitment to $\textsf{msk}$ or $\widehat{g}_0$.
    

**NOTE**: Since these steps are independent of the witness polynomial, they can be preprocessed.

## Zero Knowledge WHIR as PCS

The prover ($\mathsf{P}$) has a $\mu$-variate multilinear witness polynomial $f(x_1,\cdots, x_\mu)$ and the Verifier ($\mathsf{V}$) has an evaluation point $\vec{a} := (a_1,\cdots, a_\mu) \in \mathbb{F}_q^\mu$. The prover claims that 

$$
F = f(a_1,\cdots,a_\mu)
$$

The follow protocol allows the verifier to evaluate this claim as follows:

1. **Commitment**: The Prover ($\mathsf{P}$) computes following polynomial 
    
    $$
    \hat{f}(x_1,\cdots, x_\mu) := f(x_1,\cdots, x_\mu) + \textsf{msk}(x_1,\cdots, x_\ell)
    $$
    
    where $\textsf{msk}(x_1,\cdots,x_\ell)$ is same as $\textsf{msk}(y_1,\cdots,y_\ell)$ under basis change $y_i \mapsto x_i$ and treated as an element of $\mathbb{F}_q^{\preceq 1}[x_1,\cdots,x_\ell,x_{\ell+1},\cdots,x_\mu]$.
    
    - $\textsf{P}\longrightarrow \textsf{V}$: Prover sends
        
        $$
        [[\widehat{f}]] = \textsf{commit}(\widehat{f})
        $$
        
        to the verifier.
        
2. **Opening**: The Verifier has the evaluation point $\vec{a} = (a_1,\cdots,a_\mu)$ and samples a random field element $\beta \xleftarrow{\$} \mathbb{F}_q$; then proceeds as follows:
    - $\mathsf{V} \longrightarrow \mathsf{P}$: Verifier sends $(\beta, \vec{a})$ to the Prover. The Verifier expects the Prover to construct the following ***blinding polynomial***
        
        $$
        g(X) = g_0(X) + \sum_{j=1}^{\mu} \beta^j\cdot X^{2^{j-1}} g_j(X)
        $$
        
        and send $F := f(\vec{a})$ and $G = \widehat{g}(\vec{a})$, where $\widehat{g}(\vec{x})$ is the $\mu$-variate representation of $g(X)$, i.e.,
        
        $$
        \widehat{g}(x_1,\cdots, x_\mu) = \mathsf{U}_{\mu, X}^{-1}(g)
        $$
        
        Notice that each univariate polynomial $g_j(X)$ has maximum degree $2^{\ell} - 1$, therefore the maximum degree of $g(X)$ is $2^{\mu - 1} + 2^\ell - 1$, which is less than $2^\mu -1$ for all $\ell < \mu$. Therefore $g(X)$ can be faithfully represented in multilinear form using a $\mu$-variate polynomial. (If $\mu \leq \ell$, Prover should follow zkWHIR-1.0 scheme.)
        
    - $\mathsf{P} \longrightarrow \mathsf{V}$: On receiving $(\beta, \vec{a})$, an honest prover computes $F = f(\vec{a})$  and $G = \widehat{g}(\vec{a})$ and sends $(F, G)$ to the verifier.
3. **Proof Verification**: After receiving $(F, G)$, the protocol starts verification as follows:
    - $\textsf{V} \longrightarrow \textsf{P}$: Verifier performs following operations:
        - Samples a **non-zero** random challenge $\rho \xleftarrow{\$} \mathbb{F}_q$ and sends $\rho$ to the Prover. The Verifier expects the Prover to prove the following claim using Sumcheck:
            
            $$
            \rho\cdot F + G = \rho\cdot f(\vec{a}) + \widehat{g}(\vec{a}) = \sum_{\vec{y}\in\{0,1\}^\mu} \left [\rho\cdot f(\vec{y}) + \widehat{g}(\vec{y}) \right ]\cdot \textsf{eq}(\vec{a}, \vec{y})
            $$
            
    - $\mathsf{P} \longrightarrow \mathsf{V}$: On receiving $\rho$, an honest Prover performs the following operations:
        - Computes $[\rho\cdot f(\vec{y}) + \widehat{g}(\vec{y})]\cdot \mathsf{eq}(\vec{a}, \vec{y})$ and starts the Sumcheck protocol of non-ZK WHIR protocol.
            
            $\mathsf{P}$ *should not* recompute $[[\rho\cdot f + \widehat{g}]]$ oracle as it will be *virtually* computed from $[[f + \textsf{msk}]]$, $[[M]]$ and $[[\widehat{g}_i]]$s and Verifier’s randomness $\rho$ and $\beta$.
            
    - $\mathsf{P} \longleftrightarrow \mathsf{V}$:  Prover and Verifier perform the first $s$-rounds of Sumcheck, where $k = 2^s$ is the WHIR folding factor.
    - $\mathsf{P} \longrightarrow \mathsf{V}$: The Prover sends the first folded oracle
        
        $$
        [[H]] := \textsf{fold}_k(\rho\cdot f + g,\; \vec{r}_1)
        $$
        
        where  $\vec{r}_1 \in \mathbb{F}_q^s$ is the Verifier’s random challenges sent during Sumcheck.
        
    - $\textsf{V} \longrightarrow \textsf{P}$: On receiving $[[H]]$ the Verifier should perform following operations:
        - The Verifier computes $\mathsf{q}$ FRI random query indexes $\alpha_1, \cdots, \alpha_{\mathsf{q}} \in \Omega \subseteq \mathbb{F}_q^\times$  where it wants to query $[[H]]$.
            
            Let $\Omega_{k} := \{1, \omega, \omega^2, \cdots \omega^{k-1} \}$ denote the $k$-th roots of unity. In order to compute the fold, the Verifier needs to constructs the virtual oracle $[[\rho\cdot f + g]]$ for $\mathsf{q}$ cosets of $\Omega_k$, namely for all points in the following set:
            
            $$
            \Gamma := \{ \alpha_1\cdot \Omega_k,\cdots, \alpha_{\mathsf{q}}\cdot \Omega_k \}
            $$
            
            where
            
            $$
            \alpha_i\cdot \Omega_k := \{\alpha_i,\;\alpha_i\cdot\omega,\; \cdots\;,\;\alpha_i\cdot\omega^{k-1} \}
            $$
            
        - The Verifier samples two field elements $\tau_1, \tau_2 \xleftarrow{\$} \mathbb{F}_q$ that will be used by the Prover to batch opening proof of polynomials.
        - $\mathsf{V} \longrightarrow \mathsf{P}$: The Verifier sends $(\Gamma, \tau_1, \tau_2)$ to the Prover and expects Prover to evaluate  $M(\vec{y}, t)$ and $g_i(X)$ at each value of $\Gamma$ and **generate a batched non-ZK WHIR proof** of those evaluations.
            
            <aside>
            🙀
            
            In non-interactive setting the Verifier doesn’t need to send the potentially large set $\Gamma$ as it can be computed by both the prover and verifier directly. However, the Prover does need to send the evaluation of $\mu+1$ polynomials and batched opening proof for all these evaluations to the Verifier, which will increase the proof size.
            
            </aside>
            
    - $\textsf{P} \longrightarrow \textsf{V}$: On receiving $(\Gamma, \tau_1, \tau_2)$, an honest Prover computes the following:
        - for all $\gamma \in \Gamma$ compute:
            
            $$
            \begin{aligned} m(\gamma, {\color{Magenta}\rho}) &:= M\left(\textsf{pow}(\gamma), {\color{Magenta}-\rho}\right) = \widehat{g}_0(\textsf{pow}(\gamma)) {-} {\color{Magenta}\rho}\cdot\textsf{msk}(\textsf{pow}(\gamma))\\
            g_1(\gamma)  &: = \widehat{g}_1(\textsf{pow}(\gamma))\\ &\quad \vdots \\ g_\mu(\gamma)  &: = \widehat{g}_\mu(\textsf{pow}(\gamma))\end{aligned}
            $$
            
            where $\textsf{pow}(\gamma) := (\gamma^{2^0}, \gamma^{2^1}, \cdots, \gamma^{2^{\ell - 1}})$.
            
        - The Prover also computes a non-ZK batched proof $\pi(\rho, \tau_1,\tau_2)$ of evaluations of $\{m(\gamma,\rho), g_1(\gamma),\cdots, g_\mu(\gamma)\}_{\gamma \in \Gamma}$ as [described in the next section](https://www.notion.so/Zero-Knowledge-WHIR-with-poly-logarithmic-overhead-2e28614bdf8c80be8c95edaa2fd93f9f?pvs=21).
        - $\textsf{P} \longrightarrow \textsf{V}$: The Prover sends the evaluated values
            
            $$
            \Epsilon := \{m(\gamma,\rho), g_1(\gamma),\cdots, g_\mu(\gamma)\}_{\gamma \in \Gamma}
            $$
            
            along with the batched WHIR proof $\pi(\rho, \tau_1, \tau_2)$ to the Verifier.
            
    - $\mathsf{V} \longrightarrow \mathsf{P}$:  On receiving $(\Epsilon, \pi(\rho,\tau_1,\tau_2))$, the Verifier validates $\pi(\rho, \tau_1, \tau_2)$ against the evaluated points $\Epsilon$ and proceeds to construct the Virtual oracle $[[\rho \cdot f + g]]$ as follows. 
    
    Let $[[L]]$ be defined as follows:
        
        $$
        [[L]] := \rho\cdot\underbrace{[[f +\mathsf{msk}]]}_{\widehat{f}(\gamma)} +  \underbrace{[[\widehat{g}_0-\rho\cdot\textsf{msk}]]}_{m(\gamma, \rho)} + \sum_{i=1}^\mu \beta^i\cdot \gamma^{2^{i-1}}\cdot\underbrace{[[\widehat{g}_i]]}_{g_i(\gamma)}
        $$
        
        If the Prover is honest, then
        
        $$
        \forall \gamma \in \Gamma:\quad [[L]]  =  [[\rho\cdot f + g]]
        $$
        
        Since the values of $\rho$, $\beta$, and $\gamma$ were selected by the Verifier, it can compute the Virtual oracle $[[L]]$ from $\Epsilon$. The Verifier can then locally compute $\textsf{fold}([[L]], \vec{r}_1)$ and compare it against $[[H]]$.
        
        <aside>
        📌
        
        Note that the binding property of $[[f + \textsf{msk}]]$, $[[M]]$ and $[[\widehat{g}_i]]$s guarantee that for any given $\gamma \in \Gamma$, the value of $[[L]]$ constructed from malicious committed values of $[[f + \textsf{msk}]], [[M]]$, and $[[\widehat{g}]]$ will equal to the value of $[[\rho\cdot f + g]]$ at $\gamma$, has a maximum probability of $\frac{\mu + 1}{|\mathbb{F}_q|}$. (This follows from Schwartz–Zippel ****Lemma by treating the committed values as constants, and $\beta$ and $\rho$ as variables).
        
        Furthermore, a malicious prover that does not commit to $f(\vec{x}) + \textsf{msk}(\vec{x})$ and $\widehat{g}_0(\vec{x}) + t\cdot \textsf{msk}(\vec{x})$ as described in the protocol will be able to compute $[[\rho\cdot f + \widehat{g}_0]]$  from $[[f + \textsf{msk}]]$ and $[[M]]$ for all values of $\gamma \in \Gamma$ with probability no greater than$\frac{|\Gamma|}{|\mathbb{F}_q|}$.
        
        </aside>
        
    - Once the virtual oracle is validated against $[[H]]$, the protocol should follow WHIR as usual.

## Soundness

If the Verifier accepts, we want to establish that $F$ must be the evaluation of $f(\vec{x})$ with high probability, given commitments to $[[\widehat{f}]], [[M]], [[g_1]],\cdots, [[g_\mu]]$. We first establish that the degree of $f(\vec{x})$ must be less than $2^\mu$.

The virtual oracle constructed above is a random linear combination of some claimed at max $2^\mu - 1$ degree polynomial. Therefore, if the Verifier accepts, then by proximity gaps $f(\vec{x}) + \textsf{msk}(\vec{x})$ must be a polynomial of degree at most $2^\mu - 1$. We next establish that degree of $\textsf{msk}$ can be at most $2^{\ell + 1}-1$, therefore, if $\ell + 1 < \mu$, then the degree of $f(\vec{x}) + \textsf{msk}(\vec{x})$ must be the same as the degree of $f(\vec{x})$.

Since $[[\widehat{g}_0 - \rho\cdot\textsf{msk}]]$ passes batch evaluation proofs, the degree of $\widehat{g}_0(\vec{x}) + \rho\cdot \textsf{msk}(\vec{x})$ must be at most $2^{\ell+1}-1$. However, $\rho$ was selected by the Verifier, therefore the probability with which $\widehat{g}_0(\vec{x})$ and $\textsf{msk}(\vec{x})$ would have had degree higher than $2^{\ell+1}-1$, but their combination $\widehat{g}_0(\vec{x}) - \rho\cdot \textsf{msk}(\vec{x})$ happened to be of degree  $2^{\ell+1}-1$ can only happen with probability $\frac{1}{|\mathbb{F}_q|}$. Therefore, with high probability, *both* $\widehat{g}_0(\vec{x})$ and $\textsf{msk}(\vec{x})$ must be of degree at most $2^{\ell+1}-1$. By assumption $2^{\ell + 1} < 2^{\mu}$, so the degree of $f(\vec{x}) + \textsf{msk}(\vec{x})$ is determined by the degree of $f(\vec{x})$. Similarly, it can be argued that degree of $g(X)$ is less than $2^\mu - 1$, therefore, the virtual oracle $[[L]]$ is (close to) a Reed Solomon codeword for $\rho\cdot f(\vec{x}) + g(x)$ and by soundness of Sumcheck, $f(\vec{a}) = F$ with high probability.

## Zero Knowledge

Since each polynomial is opened at less than its degree number of points, the evaluated values are uniformly random, and can be perfectly simulated. Furthermore, given oracles $[[f + \textsf{msk}]]$ and $[[\widehat{g}_0 - t\cdot \textsf{msk}]]$, its not possible to compute the evaluation of $f(\vec{x})$ at ***any*** point from these two oracles. This is because $f$, $\widehat{g}_0$ and $\textsf{msk}$ are three independent polynomial, but each of these oracles only provide two evaluation points making it impossible to solve any evaluation of$f(\vec{x})$. 

With high probability, the blinding polynomial $g(X)$ contains non-zero coefficients for $X^{2^{i-1}}$ for all $1 \leq i \leq \mu$, therefore $\widetilde{g}(\vec{x}) = a_0 + a_1x_1 + \cdots + a_\mu x_\mu + ...$ and each Sumcheck round polynomial 

$$
\begin{aligned}
u_i(X) &:= \sum_{\vec{b} \in \{0,1\}^{\mu - i}} \left[\rho\cdot f(\vec{r}_{i-1},X, \vec{b}) + \widehat{g}(\vec{r}_{i-1}, X, \vec{b})\right]\cdot W(\vec{r}_{i-1},X, \vec{b}) \\ &\;= \sum_{\vec{b} \in \{0,1\}^{\mu - i}} \left[\rho\cdot (f_0(\vec{r}_1, \vec{b}) + X\cdot f_1(\vec{r}_1, \vec{b})) + c_0(\vec{r}_1, \vec{b}) + X\cdot c_1(\vec{r}_1, \vec{b}))\right]\cdot W(\cdots)\end{aligned}
$$

where some $c_0, c_1 \in \mathbb{F}_q^{\mu - 1}$ are some polynomials whose coefficients are guaranteed to be uniformly random because of the random linear terms in $\widetilde{g}(\vec{x})$. Therefore, a ZK simulator can perfectly simulate the sumcheck rounds polynomials. The Simulator works as follows:

- **Simulator Input**: A claim that for some $\mu$-variate polynomial $f(\vec{a}) = F$, where $2^\mu - 1$ is greater than the query complexity of FRI. The Verifier needs to simulate
    - the opened query points of FRI oracle,
    - the Sumcheck round polynomials,
    - the Sumcheck final round polynomials,
    - the opened query points of intermediate FRI oracles
- Simulator picks a random $\mu$-variate polynomial $\tilde{f}(\vec{x})$ such that $\tilde{f}(\vec{a})=F$ and number of non-zero coefficients of $\tilde{f}(\vec{x})$ is greater than the total query complexity of FRI, including all intermediate rounds. (Since FRI’s total query complexity is polylogarithmic in its univariate degree, this requirement is easily satisfied by picking a dense polynomial $\tilde{f}(\vec{x})$ such that $\tilde{f}(\vec{a}) = F$. Since the number of opened points from this oracle is less than the degree, the evaluated points of $\tilde{f}(\vec{a})$ are uniformly and identically distributed as the evaluated points of $f(\vec{x}) + \textsf{msk}(\vec{x})$.
- As discussed before, the Sumcheck round polynomials are masked using an uniformly random linear polynomial, therefore, the distribution of sumcheck round polynomials of $[\rho\cdot \tilde{f}(\vec{x}) + g(\vec{x})]W(\vec{x})$ is identially distributed as $[\rho\cdot f(\vec{x}) + g(\vec{x})]W(\vec{x})$ regardless of $W(\vec{x})$.

### Batched Proof Computation

Prover computes the batched proof as follows. Compute the *batched polynomial*

$$
S(\vec{z}, t) := M(\vec{z}, t) + \sum_{i=1}^\mu \tau_1^i\cdot \widehat{g}(\vec{z})
$$

and batched $\textsf{eq}$ polynomial 

$$
\textsf{beq}(\vec{z}, t) := \textsf{eq}(\rho, t)\cdot\left [\sum_{\gamma \in \Gamma;\; i \in [k\cdot\mathsf{q}]} \tau_2^i\cdot \textsf{eq}(\textsf{pow}(\gamma), \vec{z}) \right]
$$

and then computes the WHIR proof $\pi(\rho, \tau_1,\tau_2)$ of the claim:

$$
\sum_{\gamma \in \Gamma;\;i\in[k\cdot \mathsf{q}]} \tau_2^i\cdot m(\gamma, \rho) + {\color{Magenta} 2}\cdot\sum_{\gamma \in \Gamma;\;i\in[k\cdot \mathsf{q}]} \tau_2^i\cdot \left(\sum_{j=1}^\mu \tau_1^j\cdot g_j(\gamma)\right)\\ = \\\sum_{\vec{b} \in \{0,1\}^{\ell},\;c\in \{0,1\}}  S(\vec{b}, c)\cdot \textsf{beq}(\vec{b}, c)
$$

The factor ${\color{Magenta} 2}$ in the claim above is because each $g_i$ is an $\ell$-variate polynomial, while the sumcheck is being computed over $\ell+1$ variables.
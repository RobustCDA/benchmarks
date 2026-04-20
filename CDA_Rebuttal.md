## **`Question 1:` KZG commitment has no transparent setup, but assumes either a trusted setup or a multi-party protocol.**

KZG requires a structured reference string (SRS), which can be generated either via a trusted setup or a multi-party computation (MPC) - “Constant-Size Commitments to Polynomials and Their Applications”.
In practice, modern deployments mitigate the trust assumption through large-scale MPC ceremonies such as **Powers-of-Tau**, where security holds as long as at least one participant behaves honestly and discards their secret contribution.

This approach has been widely adopted in production systems, most notably in the Ethereum KZG ceremony used for EIP-4844. Therefore, while KZG is not transparently setup-free, the MPC-based setup significantly reduces the trust assumption and is considered practical for large-scale deployments.

---

## **`Question 2:` Verifiability too hand-wavy, e.g "Consequently, a verifier can check the RLNC piece (g, D[r, c], τ ∗ ) directly against com∗ using the standard commitment verification algorithm, without reconstructing the original symbol. Because KZG is linear and position-wise verifiable, these derived commitments and openings preserve both properties of code binding and position binding as an ECC scheme. The security of CDA thus follows."**

Since KZG is homomorphic in nature, which is cited as “Constant-Size Commitments to Polynomials and
Their Applications” paper.

On the paper they used:

- $\phi_i(x)$ for polynomial which is data for creating commitments
- $C_{\phi_i}$ for commitment of $\phi_i(x)$
- $w_{\phi_i}$ for openning

On our paper, there are a bit modification:

- $\phi_i(x)$ ⇒ $x_i$
- $C_{\phi_i}$ ⇒ $CC.Com(x_i)$, note that: $CC.Com(g_i x_i) = g_i CC.Com(x_i)$
- $w_{\phi_i}$ ⇒ $\tau_i$

In KZG paper, they have:

If $\phi(x) = \phi_1(x) + \phi_2(x)$, then they can be verified with $C_{\phi} = C_{\phi_1}C_{\phi_2}$ and $w_{\phi} = w_{\phi_1}w_{\phi_2}$

Here our paper is the same, but with different notations:

- $D[r,c] = \hat m'[r,c] =∑g_i x_i$
- $C_{\phi} = \prod C_{\phi_i}$ ⇒ $com^* = \sum g_i com_i$
- $w_{\phi} = \prod w_{\phi_i}$ ⇒ $\tau^* = \sum g_i \tau_i$

Here we have $D[r,c]$ can be verified by $com^*$ and $\tau^*$. Just to clarify, the reason we use summation $\sum$ instead of product $\prod$ is:

“Although data combination uses a summation in the field (RLNC), the corresponding commitments combine multiplicatively in the group.”

We have:

$CC.Com(x_i) = g^{\phi_i(\alpha)}$

Then,

$CC.Com(x_i)^{g_i} = (g^{\phi_i(\alpha)})^{g_i}$

Besides, KZG commitments are additively homomorphic, we have

$CC.Com(∑g_i x_i)=∏CC.Com(x_i)^{g_i} = \prod (g^{\phi_i(\alpha)})^{g_i} = g^{\sum g_i \phi_i(\alpha)}$

---

## **`Question 3:` The results show that each column consistently contains significantly more than one honest node; at least 12 honest nodes per column in both CDA and RDA.” How does this result affect the fragment size k used in RLNC?**

On the question 3, if we tune the parameter, we will realize that the result $N_{hon}$ = 3000 works with $k = 12$ with ~99.9%.

The observation that each column consistently contains at least 12 honest nodes directly justifies our choice of the RLNC fragment size, _k_ = 8.

In our RLNC scheme, a raw data column is encoded into 8 independent pieces, meaning a client needs exactly _k_ = 8 valid pieces to successfully decode the original data.

Since each node assigned to a column holds at least one encoded piece, guaranteeing 12 or more honest nodes per column ensures the network always possesses at least 12 honest, independent fragments.

This creates a critical structural safety margin (12 > 8). In engineer side, we can reduce the need of honest nodes by increasing the amount of pieces that they have to store.

---

## **`Question 4:` Nodes can behave Byzantine, but there must be at least one honest node in every column; this is achieved when a sufficient Ω(k1k2) number of good nodes join the network.” Why is this the case? Where does this result come from?**

The result follows from a standard **balls-into-bins occupancy argument**. If $N_{\mathrm{hon}}$ honest nodes join independently and each selects one of the $k_2$ columns uniformly at random, then for any fixed column c.

$X_c \sim \mathrm{Binomial}\!\left(N_{\mathrm{hon}},\frac{1}{k_2}\right),
\qquad
\mu=\mathbb{E}[X_c]=\frac{N_{\mathrm{hon}}}{k_2}.$

To guarantee at least k honest nodes in every column, we bound the bad event $X_c<k$.
Writing $\mu=(1+\alpha)k$ for some $\alpha>0$, then

$\delta
= 1 - \frac{k}{\mu}
= 1 - \frac{1}{1+\alpha}
= \frac{\alpha}{1+\alpha}.$ Therefore, the Chern-off lower-tail bound gives

$\Pr[X_c<k]
\le
e^{-\frac{\alpha^2}{2(1+\alpha)^2}\cdot \frac{N_{\mathrm{hon}}}{k_2}}.$

Applying a union bound over all $k_2$ columns and $L$ anchored windows yields

$\Pr\!\left[\exists j<L,\exists c\in[k_2]:X_c(j)<k\right]
\le
Lk_2\,e^{-\frac{\alpha^2}{2(1+\alpha)^2}\cdot \frac{N_{\mathrm{hon}}}{k_2}}.$

Equivalently,

$\Pr\!\left[\forall j<L,\forall c\in[k_2]:X_c(j)\ge k\right]
\ge
1-
Lk_2\,e^{-\frac{\alpha^2}{2(1+\alpha)^2}\cdot \frac{N_{\mathrm{hon}}}{k_2}}.$

For example, consider $k_2=64$ columns and $N_{\mathrm{hon}}=3000$ honest nodes. The expected number of honest nodes per column is

$\mu=\frac{3000}{64}=46.875.$ Setting the threshold to k=12 gives

$\delta = 1-\frac{12}{46.875}=0.744.$

By the Chern-off bound,

$\Pr[X_c<12]\le e^{-\frac{\delta^2}{2}\mu}\approx 2.32\times10^{-6}.$ Applying a union bound over all columns yields

$\Pr[\forall c\in[64]:X_c\ge12]\ge 0.99985.$

Thus, with high probability each column contains at least 12 honest nodes, supporting the choice of RLNC fragment size k=12.

---

## **`Question 5:` In Section V-C it is not clear how you calculate the commitment size.**

In the KZG, a commitment is represented by a single element of an elliptic-curve group, and therefore has constant size independent of the polynomial degree \***\*or the number of committed values**.\*\*

Consequently, the commitment size does not depend on the size of the encoded cell in our scheme. The exact byte size depends on the concrete curve and encoding used in a specific implementation.

For example, when instantiated over the **BLS12-381 curve**, a compressed G1 element occupies **48 bytes**. We will revise Section V-C to clarify that the commitment corresponds to **one group element of constant size**.

---

## **`Question 6:` Nit picks**

Page 2. “Because a chunk has to be broadcast to…” -> “Because a chunk has to be broadcasted to…” ⇒ Yes, our mistake!

Page 3. “(com, St) ← [CC.Com](http://cc.com/)(ck, m)” -> “(com, St) ← [CC.Com](http://cc.com/)(ck, \hat{m})” ⇒ Yes, PeerDAS paper wrote (com, St) ← [CC.Com](http://cc.com/)(ck, m) but they explained the step CC.Com(ck, m) included step erasure m to \hat{m}, and “(com, St) ← [CC.Com](http://cc.com/)(ck, \hat{m})” directly would be better

Page 4. Symbol “d_i” first appears in the equations. ⇒ Our mistake, d_i was in the initial version containing the detail explanation, it should be x_i and the same with “D[r, c]” on Page 5 ⇒ \hat{m}’[r,c]

---

## **`Question 7:` Why is Commitment added in CDA and what value it brings to RDA?**

RDA assumes that each symbol in the encoded block is verifiable through an underlying erasure-code commitment scheme, as in existing DAS designs such as PeerDAS.

In CDA, however, we further partition each symbol into smaller pieces and store RLNC-coded combinations of those pieces. This additional coding layer requires ensuring that the resulting coded symbols remain verifiable with respect to the original block commitment.

Therefore, we introduce the commitment construction in CDA to guarantee that symbol partitioning and RLNC coding preserve the verifiability property assumed by RDA.

---

## **`Question 8:` Analysis of the correctness and soundness of the protocol will be useful.**

Overall, correctness and soundness of CDA is exactly similar to correctness and soundness of DAS, because we keep flow a block data will be extended and nodes will do sampling.
For that properties, the rebuttal can be checked on paper namely “Foundation of Data Availability Sampling”.

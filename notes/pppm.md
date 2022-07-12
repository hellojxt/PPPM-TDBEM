Single Layer Potential
$$
V(s, x) = \int_{\Gamma}\frac{e^{-s|x-y|}}{4 \pi|x-y|} \varphi(y) d \Gamma_{y}
$$
Double Layer Potential
$$
D(s, x) = \int_{\Gamma} \frac{\partial}{\partial \nu_{y}} \frac{e^{-s|x-y|}}{4 \pi|x-y|} \varphi(y) d \Gamma_{y}
$$

Solve Equation
$$
\begin{aligned}
    \frac{1}{2} G_{n}(x) &= &\sum_{j=0}^{n} \sum_{l = 0}^m \left(-W_{n-j}^{\Delta t}(\mathcal{V_l})  \Phi_{j,l}+ W_{n-j}^{\Delta t}(\mathcal{D_l}) G_{j,l}\right) \\
& = &\sum_{j=0}^{n} \sum_{l = 0}^{m_0 - 1} \left(-W_{n-j}^{\Delta t}(\mathcal{V_l})  \Phi_{j,l}+ W_{n-j}^{\Delta t}(\mathcal{D_l}) G_{j,l}\right) \\
&& +\sum_{j=0}^{n} \sum_{l = m_0}^m \left(-W_{n-j}^{\Delta t}(\mathcal{V_l})  \Phi_{j,l}+ W_{n-j}^{\Delta t}(\mathcal{D_l}) G_{j,l}\right)\\ 
\end{aligned} 

$$
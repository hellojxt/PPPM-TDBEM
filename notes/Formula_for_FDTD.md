## FDTD公式推导
$$
p_{i, j, k}^{n+1} =\frac{\lambda^{3}(p_{i, j-1, k}^{n}+p_{i, j+1, k}^{n}+p_{i, j, k-1}^{n}+p_{i, j, k+1}^{n}-4p_{i, j, k}^{n})+\lambda ^2(p_{i, j-1, k}^{n}+p_{i, j+1, k}^{n}+p_{i, j, k-1}^{n}+p_{i, j, k+1}^{n}-p_{i-1, j, k}^{n-1}-p_{i+1, j, k}^{n-1}+2p_{i-1, j, k}^{n}-6p_{i, j, k}^{n})+\lambda (4p_{i, j, k}^{n}-2p_{i, j, k}^{n-1})+2p_{i, j, k}^{n}-p_{i, j, k}^{n-1}}{1+2 \lambda}
$$

$$
p_{i+1, j, k}^{n} =\frac{\lambda(8p_{i, j, k}^{n}-p_{i, j-1, k}^{n}-p_{i, j+1, k}^{n}-p_{i, j, k-1}^{n}-p_{i, j, k+1}^{n}-2p_{i-1, j, k}^{n})+p_{i-1, j, k}^{n}-p_{i-1, j, k}^{n-1}+p_{i+1, j, k}^{n-1}}{1+2 \lambda}
$$

其中$\lambda=\frac{c\tau}{h}$
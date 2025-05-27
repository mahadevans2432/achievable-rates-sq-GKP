# Achievable rates for concatenated square Gottesman-Kitaev-Preskill codes
This repository hosts the source code and data accompanying the manuscript [arXiv:2505.10499 [quant-ph]](https://arxiv.org/abs/2505.10499v1).

The source code to generate all the data in Figures 1(c), 2(a), 2(c), 10(a), and 10(b) can be found in all_plots.ipynb jupyter notebook. The remaining plots use data generated through monte-carlo simulations using functions defined in monte_carlo_funcs.py (methods described in manuscript). All data generated through this method is for estimation of Bhattachary parameters and is contained in the folders Zdata and Zdata_alpha. Kindly refer to the manuscript for understanding appropriate notation.

File name legend:
- "I_analog_sq_d_2-17.npy" contains a list of $I^{\mathrm{sq}}_{d,\mathrm{analog}}$ for prime number values of $d=2$ to $d=17$. Each entry of this list is itself a list of 1000 values corresponding to this function evaluated at $\sigma=0.1$ to $0.62$ at 1000 equally spaced apart values (including both boundaries).
- "I_no_analog_sq_d_2-17.npy" similarly contains a list of $I^{\mathrm{sq}}_{d,\mathrm{no\text{ }analog}}$ for prime number values of $d=2$ to $d=17$ evaluated over the same $\sigma$ values.
- "I_analog_hexd2_0.32to0.62.npy" contains a list of the values of $I^{\mathrm{hex}}_{2,\mathrm{analog}}$ evaluated at $\sigma=0.32$ to $0.62$ at 1000 equally spaced apart values (including both boundaries).
- In Zdata: kernels are set to default of $\alpha=1$. Files named as "Zwi_sig(noise strength $\sigma$)_(blocklength $N$)_d(qudit dimension).npy" are list of bhattacharya parameters $Z(W^{(i)}_1)$ (indexed by $i$) corresponding to noise in $\hat{q}$ quadrature and file name with "Zwi_sig(noise strength $\sigma$)_(blocklength $N$)_d(qudit dimension)_P.npy" are similarly for $Z(W^{(i)}_2)$ correspondingh to $\hat{p}$ quadrature.
- In Zdata_alpha: file names are "Zwi_sig(noise strength $\sigma$)_(blocklength $N$)_d(qudit dimension)_a(value of $\alpha$)_(Q or P depending on quadrature).npy". Indexing follows similarly as Zdata.

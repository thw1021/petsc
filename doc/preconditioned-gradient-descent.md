---
orphan: true
---


# Notes on preconditioned gradient descent

:::{admonition} Jed Brown

For linear solvers, "changing the inner product" is known as "preconditioning". I wonder if that would be a fruitful way to convey the semantic also for optimization.

:::

## Preconditioned Richardson -- via classical splitting

If one wishes to solve

$$
A x + B x - b = 0
$$
and has an efficient way to solve with $ A $, then one can create an iterative scheme by solving

$$
A x^{n+1} + Bx^n - b = 0
$$
for $x^{n+1}$.
Which can be rewritten as

$$
x^{n+1} = - A^{-1}(Bx^n - b)
$$
or in defect-correction form

$$
x^{n+1} = x^n - A^{-1}(Ax^n + Bx^n - b).
$$
It is useful to recognize the error propagation in preconditioned Richardson.

$$
e^{n+1} = (I - A^{-1}(A + B))e^n.
$$

$$
||e^{n+1}||^2 = \rho(I - A^{-1}(A + B)) ||e^n|| = \rho(A^{-1}B) ||e^n||^2
$$
so the convergence (or lack there of) is determined by the eigenvalues of $ \rho(A^{-1}B).$


## Linearly preconditioned nonlinear Richardson

Similarly,

$$
F(x) = A x + \alpha(x) - b = 0
$$
can be solved with the iteration

$$
A x^{n+1} + \alpha(x^n) - b = 0.
$$
Which can be rewritten in  in defect-correction form as

$$
x^{n+1} = x^n - A^{-1}(Ax^n + \alpha(x^n) - b) = x^n - A^{-1}F(x^n).
$$

It is also possible to use a nonlinear solver to nonlinearly precondition a nonlinear system, {cite}`bruneknepleysmithtu15`, but that is not needed for this discussion.
`SNES` provides access to this general nonlinear preconditioner with `SNESGetNPC(SNES,SNES*)`.

It is not possible to write an equation for the error propagation but one can see if $A$ dominates $\alpha(x)$ one might expect good convergence otherwise
there is no reason to even expect convergence.

## Preconditioning gradient descent

The solution to

$$
\min f(x) = \min x^TA x + \beta(x) - b^Tx.
$$
is the nonlinear equation

$$
    A x + \beta'(x) - b
$$
whose preconditioned nonlinear Richardson iteration is

$$
x^{n+1} = x^n - A^{-1}(Ax^n + \beta'(x^n) - b) = x^n - A^{-1}f'(x^n).
$$

Of course, to expect the iteration to converge at all, or fast, the operator $A$ has to **dominate**
 $\beta'(x^n)$.


:::{admonition} Finite element example

Consider the minimization of the function $u$

$$
\min |u|^2_{H_1(\Omega)} + \beta(u) - (b,u)_{L^2(\Omega)}
$$

$$
\min \int_{\Omega} |u'(x)|^2 dx + \beta(u) - \int_{\Omega} b(x) u(x) dx.
$$
Using a piecewise linear finite element discretization $ u(x) = \sum u_i \phi(x) $, and for simplicity expressing $ b(x) = \sum b_i \phi(x)$, 
results in the finite dimensional optimization problem

$$
\min \int_{\Omega} (\sum_i u_i \phi_i'(x))^2 dx + \beta(\sum_i u_i \phi_i(x)) - \int_{\Omega} b(x) \sum_i u_i \phi_i(x) dx
$$

$$
\min \hat{u}^T K \hat{u}  + \hat{\beta}(\hat{u}) - \hat{b}^{T}\hat{u}.
$$
Applying the preconditioning gradient descent method produces

$$
\hat{u}^{n+1} = \hat{u}^n - K^{-1}(K\hat{u}^n + \hat{\beta}'(\hat{u}^n) - \hat{b}) =  \hat{u}^n - K^{-1} f'(\hat{u}).
$$

:::

:::{admonition} Another finite element example
Consider a more general bilinear form $a(x,u,v)$ and the minimization

$$
\min \int_{\Omega} a(x,u'(x),u'(x)) dx + \beta(u) - \int_{\Omega} b(x) u(x) dx.
$$

$$
\min \int_{\Omega} a(x,\sum_i u_i \phi_i'(x),\sum_i u_i \phi_i'(x)) dx + \beta(\sum_i u_i \phi_i(x)) - \int_{\Omega} b(x) \sum_i u_i \phi_i(x) dx
$$

$$
\min \hat{u}^T K_a \hat{u}  + \hat{\beta}(\hat{u}) - \hat{b}^{T}\hat{u}.
$$
Applying the preconditioning gradient descent method produces

$$
\hat{u}^{n+1} = \hat{u}^n - K_a^{-1}(K_a\hat{u}^n + \hat{\beta}'(\hat{u}^n) - \hat{b}) =  \hat{u}^n - K_a^{-1} f'(\hat{u}).
$$

But say we don't know how to solve with $ K_a $? If

$$
c |u|^2_{H^1(\Omega)} \le \int_{\Omega} a(x,u'(x),u'(x)) dx \le C |u|^2_{H^1(\Omega)}
$$
then we can consider replacing the preconditioner $K_a$ with $K$.

$$
\hat{u}^{n+1} = \hat{u}^n - K^{-1}(K_a\hat{u}^n + \hat{\beta}'(\hat{u}^n) - \hat{b}) =  \hat{u}^n - K^{-1} f'(\hat{u}).
$$

Easy-peasy.
:::

## Preconditioning Tao solvers

For preconditioning `Tao` solvers, as Matt has requested, I propose an interface in analogy with `KSPGetPC()` and `SNESGetNPC()` of 

```
TaoGetOPC(Tao, KSP *)
```

where the `O` is reminder this is a preconditioner for optimization. Then all Matt needs to do in Firedrake is

```
TaoGetOPC(tao, &ksp);
KSPSetOperators(ksp,K,Kp);
```

Since applications of both $K$ and $K^-1$ may be needed in the preconditioned Tao algorithms perhaps $K $ and $K_p$ should be provided directly, with for example,

```
TaoSetLinearPreconditioner(Tao, K, Kp)
```

or

```
TaoSetLinearPreconditionerOperators(Tao, K, Kp)
```


but I'm not sure if the language usage is clear.


Note that Matt has only requested the concept of "linear" preconditioning in Tao so I do not, nor could, discuss how to extend to nonlinear preconditioning in Tao.
I leave that for another decade.


## The Reisz map business

:::{admonition} What the heck are Matt and Toby talking about?

Functional analysis is the equivalent of linear algebra but for function spaces instead of $R^N$, so some terms from functional analysis are useful to
understanding solvers/optimizers when used with finite element approximations of function spaces.
In fact, some groups develop their algorithms directly in function spaces and then (only while writing the code) convert them
to linear algebra expressions. This is not the model used in PETSc. Functional analysis is very powerful, but the community that speaks it fluently
(and most importantly totally correctly) is
small hence PETSc attempts to require as little knowledge of functional analysis as possible from its users and developers.
Plus PETSc/TAO is not just for the PDE community and though we strive
to provide as much support for PDE problems as possible we also try to avoid using limiting functional analysis/PDE-ish language in the API when not necessary.

:::


So what does the Hilbert space, the inner product, the dual space, and the Reisz map have to do with anything?

Consider the generalization

$$
\min \int_{\Omega} a(x,u(x)) dx + \beta(u) - \int_{\Omega} b(x) u(x) dx.
$$

where $ a(x,u(x)) $ is, for simplicity, bilinear in combinations of derivatives of $u()$. For example, $ a(u(x))n = u''(x)u''(x) $.


How can one decide how to precondition the optimization problem assuming the first term dominates the optimization process?

The function space of $u()$ tells us what forms of the derivatives of $u()$ appear in $ a()$ which tells us what linear operator is likely
to be a good optimization preconditioner.


:::{admonition} Biharmonic example

$$
\min \int_{\Omega} u''(x)u''(x) dx + \beta(u) - \int_{\Omega} b(x) u(x) dx.
$$

$$
\min \int_{\Omega} a(x,\sum_i u_i \phi_i''(x),\sum_i u_i \phi_i''(x)) dx + \beta(\sum_i u_i \phi_i(x)) - \int_{\Omega} b(x) \sum_i u_i \phi_i(x) dx
$$

$$
\min \hat{u}^T K_b \hat{u}  + \hat{\beta}(\hat{u}) - \hat{b}^{T}\hat{u}.
$$
Applying the preconditioning gradient descent method produces

$$
\hat{u}^{n+1} = \hat{u}^n - K_b^{-1}(K_b\hat{u}^n + \hat{\beta}'(\hat{u}^n) - \hat{b}) =  \hat{u}^n - K_b^{-1} f'(\hat{u}).
$$

:::

Now say someone refused to tell me what $a()$ was, because they are being pedantic,  but they did tell me $ u \in V $ (the above example $ V = H^2(\Omega)$)?

Then I could
back out that the dominate part of $ a(u,u)$ is $ (u,u)_{V} $ whose discrete representation is $ \hat{u}^T K_V \hat{u} $ and hence I know I should
precondition with the "mass matrix" $K^-1_V$. $K^-1_V$ is the discrete realization of the Reisz map.

It is really **the dominant part of the minimization function that tells you what preconditioner you should use**.
The function space tells you the form (what derivatives it contains)
of the dominant part of the minimization function which then tells you the preconditioner you should use. Because of this I think the use of norm, inner product
or Reisz map does not belong in the Tao API for indicating the Tao preconditioner.

## More details

Let $ u, v $ be in a Hilbert space, $V$ with the inner product $ (,)_M$.
Define the directional derivative of $ f(u)$ as the linear (on $v$) operator

$$
\delta f(u) v = \lim_{\epsilon \rightarrow 0} \frac{f(u + \epsilon v) - f(u)}{\epsilon}.
$$


$ \delta f(u) v $ measures the infinitesimal change in the function value at $u$ in each direction $v$.

Recall, by definition, the duel space of $V$, denoted by $V'$, is the Hilbert space of linear functionals on elements of $V$. Hence $ \delta f(u) \in V'$.

The Reisz representation theorem states that for every $w' \in V'$ there exists a unique $w \in V$ such that $ (w,v)_M = w' v$. $w \in V$ can be called the
Reisz representation of $w' \in V'$.


## Revisiting elementary functional analysis
Many discussions of functional analysis focus on three examples $R^n$, $l_2$, and the Sobolev function spaces $W^{k,p}(\Omega)$.
The overly simple to the seriously complex. It is useful to consider another collection of spaces to provide insight to the Sobolev spaces.

Recall $l_2$ is the Hilbert space consisting of all sequences, $x_i$, such that

$$
\sum_i x_i^2 < \infty,
$$
with the inner product $ (x,y)_{l_2} = \sum_i x_i y_i$.

Define $h^j$ as the vector space consisting of all sequences, $x_i$, such that

$$
\sum_i x_i^2 i^{2j}< \infty.
$$
with the inner product $ (x,y)_{h^2} = \sum_i x_i y_i i^{2j}$. It is trivial to demonstrate $h^j$ is a Hilbert space. It is also trivial to see that

$$
... \, \supset  h^{-2} \supset h^{-1} \supset l_2  \supset  h^{1} \supset  h^{2} \supset  \, ...
$$
Consider $x_i = i^{-1}$ which satisfies $ (x,x)_{l_2} = \sum i^{-2} < \infty $ but not
$ (x,x)_{h^1} = \sum i^{-2}i^2 <  \infty. $


One can consider the inner product to be defined by a (infinite) diagonal matrix with entries $ M_{ii} = i^{2j}$. Then

$$
(x,y)_{h^j}  = (M^jx,y)_{l_2} = (M^{j/2}x,M^{j/2}y)_{l_2},
$$

$$
M^{1/2}: h^{j} \rightarrow h^{j-1},
$$

$$
M^{j/2}: h^{j} \rightarrow l_2,
$$
and

$$
M^j: h^{j} \rightarrow h^{-j}.
$$

By the Reisz representation theorem any continuous linear functional on $h^j$ can be represented as

$$
l(x) = (y,x)_{h^j}
$$
for a unique $ y \in h^j$. But

$$
(y,x)_{h^j} = (M^jy,x)_{l_2}.
$$
Thus

$$
l(x) = (z,x)_{l_2}
$$
for a unique $ z \in h^{-j}$. In other words, any continuous linear functional on $h^j$ can be represented using a unique member of $h^{-j}$
which means the continuous dual space of
$h^j$ is isomorphic to $h^{-j}$. Hence $h^{j}$ and $h^{-j}$ are said to be dual.

Note that directly by the Reisz representation theorem the continuous dual space of $h^j$ is also isomorphic to $h^j$. But see the
comments at https://math.stackexchange.com/questions/246735/dual-space-of-the-sobolev-spaces by
Paul Garrett that are way over my head.  

Note in all the construction above $j$ need not be an integer, leading to fractional $h^j$ spaces.

One can construct less trivial collections of Hilbert spaces by considering inner products defined by non-diagonal matrices. For example, the
tri-diagonal matrix $ (i-1)^{2j}, i^{2j}, (i+1)^{2j}.$

### Gradient descent in $ h^j$


stuff here


Consider diagonal linear operators, $L:h^j \rightarrow h^l$ with entries $L_{ii} = i^k $. Which generates


$$
y = L x = {x_1, x_2 2^k, x_2 3^k, ...}
$$

$$
|| y ||^2_{h^l} = \sum x_i^2 i^j i^{-j} i^k i^l \le \infty
$$
if

$$
l = j - k.
$$





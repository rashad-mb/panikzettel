#import "conf.typ": algoBox, conf, defiBox, theoBox
#import "@preview/cetz:0.2.0": canvas, plot


#show: conf.with(
  title: "Elements of Machine Learning and Data Science",
  shortTitle: "EMLDS",
  authors: (
    (name: "Rashad Mubayed"),
  ),
  lang: "en",
  filename: "male",
  showOutline: true,
)


#let Cl = $cal(C)$

= Data Science
Many descriptive statsitics techniques are used in Data Science to summarize and visualize data. Some common techniques include:
/ Measures of Central Tendency: mean, median, mode
/ Measures of Dispersion: range, variance, 
/ Data Visualization: histograms, box plots, scatter plots
/ Correlation Analysis: Pearson correlation coefficient

This was all covered in Stocha. It will not be repeated here.

== Decision Trees
- well suited for tabular data
- a decision tree is a flowchart-like structure
  - each internal node represents a "test" on a feature,
  - each branch represents the outcome of the test, 
  - each leaf node represents a class label (or a distribution over class labels).

- goal: create a model that predicts the value of a target variable based on several input features.

=== Metrics for Decision Trees  

/ Entropy of target feature t: $ H(t) = - sum_(k=1)^K (P(t=k) dot log_2 P(t=k)) $
- measure of impurity or uncertainty in a dataset
- minimal entropy is 0 (when all instances belong to the same class), maximal entropy is $log_2(K)$ (when instances are evenly distributed across K classes).

/ Overall Entropy: $ H_W^d(t) = sum_("node" in "leaf_nodes"(d)) ((|"node"|)/N dot H^("node")(t)) $
- weighted average of individual entropies 

/ Information Gain: $ "IG(d)" = H(t) - H^d_W(t) $
- how much the entropy decreases after a dataset is split on feature d
- $H^d_W(t)$ is the overall entropy after splitting on feature d
- higher IG means better feature for splitting (root of the decision tree is the target feature with the highest IG).

/ Information Gain Ratio:   $ "GR(d)" = "IG"(d)/H(d) $
- punishes feature splits that create large trees
- helps to avoid overfitting by preferring features that provide a good balance between information gain and tree complexity

=== Pruning Decision Trees
- branches are pruned when they dont improve performance on validation data.
- post-pruning: if the sum of misclassifications in child nodes is greater than in the parent node, we prune the child nodes. 
- pre-pruning: stop splitting a node if the information gain is below a certain threshold.

==== ID3 Algorithmus
- recursively build the decision tree by selecting the feature with the highest information gain at each node until all instances belong to the same class or no features are left to split on.

== Clustering
- falls in the category of unsupervised learning (obtain a model from unlabeled data, no target features).
- goal: group a set of objects in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups.

We use multiple _distance metrics_ to measure similarity between data points:
/ Euclidean Distance: $d(x,y) = sqrt((x_1 - y_1)^2 + (x_2 - y_2)^2)$
/ Manhattan: $d(x,y) = |x_1-y_1| + |x_2 - y_2|$
/ Chebychev: $d(x,y) = limits(max)_i (|x_i - y_i|)$
/ Minkowski ($L^p$): $d(x,y) = root(p, limits(sum)_i |x_i - y_i|^p)$
/ Jaccard Distance: $d(X,Y) = 1 - (|X inter Y|)/(|X union Y|)$ (for sets)

=== k-means
- 2 strong assumptions:
  - each cluster is represented by its centroid 
  - each point belongs to the cluster with the nearest centroid (w.r.t a distance metric)

- algorithm:
  1. choose k (initial) centroids (randomly or using heuristics)
  2. while not converged:
    - assign each data point to the nearest centroid
    - update centroids by calculating the geometric center of all points assigned to each cluster

- only finds circular/spherical clusters (due to centroid definition). no complex shapes.
- sensitve to outliers (can pull centroids away from true cluster center).
- quality of clusters is measured using sum of all sqaures between all instances to their closest centroid:
$ E = limits(sum)_(i=1)^k limits(sum)_(x in Cl_i) d(x, c_i)^2 $
where $c_i$ is the centroid of cluster $Cl_i$.


=== k-medoids
- centroids are now actual data points (medoids) instead of geometric centers.
- algorithm is similar to k-means:
  1. choose k initial medoids (randomly or using heuristics)
  2. while not converged:
    - assign each data point to the nearest medoid
    - for each cluster, select the data point that minimizes the sum of distances to all other points in the cluster as the new medoid


- more robust to outliers (since medoids are actual data points).
- more computationally expensive than k-means (due to distance calculations between all points in a cluster).

=== Agglomerative Clustering (Dendrograms)

- bottom up approach instead of top down 
- no initial guess of number of clusters k needed

#figure(
  image("img/male/dendrogram.png", height: 5cm),
  caption: [Beispiel eines Dendrogramms],
) <dendrogram>

1. create a cluster $Cl_i$ for each data point $x_i$
2. repeat until only one cluster remains:
  - compute distances $d(Cl_i, Cl_j)$ between all clusters
  - merge the two closest clusters $min d(Cl_i, Cl_j)$

The distance between clusters can be defined in multiple ways: minimum linkage, maximum linkage, mean linkage (using centroids), average linkage.


=== Density-Based Clustering (DBSCAN)

- 2 parameters: $epsilon$ (neighbourhood size) and _MinPts_ (minimum number of points in the radius to be a core point).
- an instance is a core point if there are at least _MinPts_ points within its $epsilon$-neighborhood.
- an instance is directly reachable from another instance if it is within its $epsilon$-neighborhood, or if it is the point itself. (transitiv, but not symmetric)
- a cluster is a set of instances where each instance is reachable from at least one core point in the cluster.
- can detect clusters of arbitrary shape 



== Frequent Itemsets
#let support = math.op("support")
#let supportCount = math.op("support_count")

Notation:
- $I={I_1, ..., I_D}$ are _items_.
- $A subset.eq I$ is an _itemset_.
- A _transaction_ $T subset.eq I$ is a non-empty itemset.
- A dataset $X = {T_1, ..., T_N}$ is a set of transactions.

Metrics:
$ support(A) = ("support_count"(A))/(|X|) = (|[T in X | A subset.eq T]|)/(|X|) $
- the support of an itemset A is the fraction of transactions in the dataset X that contain the itemset A.

Example:
$
  X & = [{A,B,E}, {C,B}, {A,D}, {A,D,B}] \
  T & = {A,B} subset.eq I
$

$supportCount(A)= |[T_1,T_4]| = 2$, thus $support(A) = 2/4 = 1/2$


- $A$ is a frequent itemset, if $support(A) >= "min_sup"$. 
- $A$ is closed, if $support(A) > support(B)$ for all $B supset A$.
- $A$ is a maximal frequent itemset, if there is no frequent superset of $A$ (closed by definition).

We have 2 main algorithms to find frequent itemsets: Apriori and FP-Growth.

=== Apriori

1. Generate all frequent itemsets of length 1: $L_1$.
2. For $k=1$ to max length:
  1. Generate candidate itemsets of length $k+1$: $C_(k+1)$ from $L_k$.
  2. Prune candidates in $C_(k+1)$ that have infrequent subsets.
  3. Count the support of each candidate in $C_(k+1)$ by scanning the dataset.
  4. Generate $L_(k+1)$ by selecting candidates from $C_(k+1)$ that meet the minimum support threshold.

==== Pruning rules:
- If $A subset.eq B$, then $support(A) >= support(B)$.
- If $A subset.eq B$ and $B$ is frequent, then $A$ is also frequent.
- If $A subset.eq B$ and $A$ is infrequent, then $B$ is also infrequent.

#figure(
  image("img/male/apriori.png", height: 13cm),
  caption: [Example of Apriori Algorithm],
)

==== Limitations:
- challenging to generate candidate itemsets for large datasets 
- requires testing of each candidate against all transactions

=== FP-Growth
- A datastructure called FP-Tree is used to store the dataset in a compressed form.
- efficient, only 2 passes through the dataset needed.

==== FP-Tree Construction:
1. Determine the frequency of each item and sort them in descending order.
3. Remove infrequent items from transactions.
4. Build the FP-Tree:
  1. Start with a null root.
  2. For each transaction, insert it into the tree, updating counts of existing nodes or creating new nodes as necessary.

To reconstruct the itemsets from the tree, simply look at the paths from the root to the leaves, combining items along the way.

==== FP-Tree Mining: 
- used to determine frequent itemsets from the FP-Tree.
- Start from the bottom of the tree and work upwards.
- Each level of the tree is seperated from the others ("altitude lines")
- For each item $X$:
  1. Assume the singleton ${X}$ itemset of the item is frequent (if it meets the min_sup).
  2. Work upwards in the tree, updating the counts of itemsets that include the item ${X}$. (conditional tree)
  3. If a higher node loses its count below min_sup, prune it and all its children.
  4. Go up to the next node. Repeat until the tree is empty.


== Association-Rules
Goal: find ARs  with high support and high confidence.

/ Association Rule: $A => B "with" A subset.eq I, B subset.eq I, A inter B = emptyset$. E. g. {Cheese, Bread} => {Milk}

/ Support : fraction of transactions containing both A and B
 $ support(A => B) = support(B => A)= support(A union B) = supportCount(A union B)/(|X|) $

/ Confidence: fraction of transactions containing A that also contain B

 $ "conf"(A => B) = supportCount(A union B)/supportCount(A) $

  - Confidence is not symmetric. 
  - Adding items to the right side makes the rule more specific, thus usually decreases confidence.
  - No clear realtion when adding items to the left side.

/ Lift: fraction of a AR's support to that expected if A and B were independent. Determines the quality of the rule.


 $ "lift"(A => B) = support(A union B)/(support(A) dot support(B)) = "conf"(A => B)/support(B) = ("#"A B dot "#ALL") / ("#" A dot "#" B) $

  - lift >> 1: strong positive correlation between A and B
  - lift $approx$ 1: A and B are independent
  - lift << 1: strong negative correlation between A and B

- only use closed frequent itemsets to generate rules (to reduce redundancy).
- redundant rules are pruned if they have a stronger rule (more items on the right side).



/ Simpons-Paradox:  A trend appears in several different groups of
data but disappears or reverses when these
groups are combined.


== Time-Series Analysis


= Machine Learning



=== Our Goal

Goal: with training data $cal(D) = {x_1,..., x_n}$ or with labels $cal(D) = {(x_1, t_1),..., (x_n, t_1)}$, learn a function $y$ that can make predictions on data not in the test set.

/ Supervised Learning: learn a function $y: cal(X) -> cal(T)$ from labeled data
/ Unsupervised Learning: learn patterns from unlabeled data (e.g. clustering)
/ Semi-Supervised Learning: learn from a mix of labeled and unlabeled data

== Probability Density Estimation

a-priori probability $ p(cal(C)_k)$: how likely is class $cal(C)_k$ to occur in the data?

Likelihood $p(x | cal(C)_k)$: how likely is it to observe a data point $x$ given an example of class $cal(C)_k$?

a-posteriopri probability $ p(cal(C)_k | x)$: what is the probability for class $cal(C)_k$ given a data point $x$?

$
  p(cal(C)_k | x) = (p(x | cal(C)_k)p(cal(C)_k))/(p(x)) = (p(x | cal(C)_k)p(cal(C)_k))/(sum_j p(x | cal(C)_j) p(cal(C)_j))
$

Intuition: $("likelihood" dot "prior") / "normalization factor"$ 

Our goal is to minimize the classification error. For that, we decide for class $cal(C)_k$ if $p(cal(C)_k | x) > p(cal(C)_j | x)$ for all $j != k$.

== Parametric Methods to Estimate Probability Densities


=== Maximum Likelihood Estimation (MLE)
We assume that the probability density function (pdf) $p(cal(C)_k | x)$ has a known functional form with unknown parameters $theta$ (in the normal distrubion case, $theta = (mu, sigma)$). Our goal is to estimate these parameters from the data $chi = {x_1,...,x_N}$

The likelihood of $theta$ is the probability that the data $chi$ was generated by the parameters $theta$.

$ 
  L(theta) = p(chi | theta) = product_(n=1)^N p(x_n | theta)
$

To maximamize $L(theta)$, we _minimze_ the log-likelihood ("Energy") $E(theta)$:
$
  E(theta) = - ln L(theta) = - sum_(n=1)^N ln p(x_n | theta)
$

For example, in the normal distribution $cal(N)(x | mu, sigma)$ we estimate $hat(theta) = (hat(mu), hat(sigma))$ by computing $(partial E(mu, sigma)) / (partial mu)  = 0$ and $(partial E(mu, sigma)) / (partial sigma)  = 0$.

The MLE $hat(sigma)$ is biased for small datasets, thus we use the _unbiased estimator_ for the variance:
$
  hat(sigma)^2 = 1/(N-1) sum_(n=1)^N (x_n - hat(mu))^2
$

== Non-Parametric Methods: Histograms, Kernel Methods & k-Nearest Neighbors

These are methods to estimate probability density functions without assuming a specific functional form (and thus no parameters to estimate).

=== Histograms

Partiton of data into N bins of width $Delta$. The height of bin $i$ is then $p_i = n_i/(N Delta_i)$, where $n_i$ is the number of observed outcomes in bin $i$. This can be done for every diemensionality D, but number of bins grows exponentially with D.

- Advantages: general method, easy to implement, no need to store all data points after constructing the histogram.

- Disadvantages: choosing bin width is non-trivial, not smooth, poor performance in high dimensions.

=== Kernel Methods 
Goal: etermine the number $K$ of data points inside a _fixed_ hypercube (kernel).

The kernel function $k(u)$ is 1, whenever the u lies in the hypercube. u is hearby a vector from $x$ to some point $x_n$. If we sum up over all data points, we get the number of data points $K$ in the kernel.

The probability desnity estinmate is: 
 $ p(x | Cl_k)  approx K/(N V) = 1/(N h^D) sum_(n=1)^N k(x-x_n) $

=== k-Nearest Neighbors

Goal: increase the volume $V$ until fixed value $K$ data points are inside the hypersphere. The probability density estimate remains:
$
  p(x | Cl_k) approx K / ( N V^*)
$

- Advantages: no computation during training (we just store the data points). Very general and can be used to represent every pdf
- Disadvantages: computationally expensive  (linear with data points), choosing the kernel size is non-trivial

#grid(
  columns: (1fr, 1fr),
  [#figure(
    image("img/male/kernel-method.png", width: 30%),
    caption: [Kernel Method],
  ) <k-nearest>],
  [#figure(
    image("img/male/k-nearest-neighbors.svg", width: 30%),
    caption: [1-Nearest Neighbors],
  ) <k-nearest>],
)

== Mixture Models

In einem Mixture Model wird davon ausgegangen, nach mehreren ($M in NN$) probability distributions verteilt zu sein, die sich addieren. Meist wird die Normalverteilung gewählt.

Also $p( x | theta ) = sum_(j=1)^M p(x|theta_j) p(j)$, wobei $theta = (pi_1, theta_1, ..., pi_1, theta_M)$ alle Parameter bündelt und $p(j)=pi_j$ die priors (also die Anfangsvermutungen des _mixtures components_).

Um hieraus eine formale Wahrscheinlichkeitsverteilung zu erstellen, muss $sum_(j=1)^M pi_j = 1$ und somit $integral p(x|theta)d x = 1$ sein.

Hierfür gibt es kein analytisches Verfahren, die Maximum Log Likelihoods zu finden. Somit bleiben numerische Verfahren (und) iterative Optimierungsverfahren.

=== EM Algorithmus
Der _E-STEP_: $ gamma_j(x_n) <- (pi_j cal(N)(x_n | mu_j, Sigma_j))/(sum_(k=1)^K pi_k cal(N)(x_n | mu_k, Sigma_k)) $
Hier ist $Sigma$ die Kovarianzmatrix.

Der _M-STEP_:
#grid(
  columns: (1fr, 1fr),
  $
       hat(N) & <- sum_(n=1)^N gamma_j(x_n) \
    hat(pi)_j & <- hat(N)_j/N
  $,
  $
    hat(mu)_j &<- 1/hat(N)_j sum_(n=1)^N gamma_j(x_n)x_n \
    hat(Sigma)_j &<- 1/hat(N)_j sum_(n=1)^N gamma_j(x_n)(x_n - hat(mu)_j)(x_n - hat(mu)_j)^sans(T)
  $,
)

EM Algorithmus muss durch _regularization_ gegen $sigma -> 0$ geschützt werden, weswegen $sigma_min I$ addiert wird #footnote("Der EM Algorithmus wird aus meiner Sicht nicht groß in der Klausur angewandt werden können, Wissensfragen jedoch schon.")

== Linear Discriminants
- a _linear discriminant_ $y(x)$ is a linear function that separates two or more classes of data points.
$ y(x) = w^sans(T) x + w_0 = sum_(i=0)^D w_i x_i $
- here we move away from probability theorie and focus on finding a decision boundary between classes.
- $y(x)$ is the distance of point $x$ from the decision boundary.

For K classes we have K linear discriminants $y_k(x) = w_k^sans(T)x + w_(k 0)$. Those can be grouped together to efficiently represent the decision boundaries between classes.

==== Least-Squares Classification
$ E(W) = 1/2sum_(n=1)^N sum_(k=1)^K (w_k^sans(T)x_n - t_(n k))^2 $
- W is the matrix of weights for all K classes
- $t_(n k)$ is 1 if data point n belongs to class k, else 0
- minimize E(W) to find the optimal weights w:
$ w = (X^sans(T)X)^(-1)X^sans(T)t = X^dagger t $ ($X^dagger$ pseudo-invers) 
- closed-form solution: $y(x;w) = w^sans(T) x = t^sans(T) (X^dagger)^sans(T) x$

==== Generalized Linear Discriminants
- outliers can ruin least-squares classification
- introduce an activation function g to make it less sensitive to outliers. 
- here: sigmoid $sigma(x) = 1/(1+e^(-x))$
- new discriminant: $y(x) = sigma(w^sans(T)x)$

==== Basis Functions
- data that is not linearly separable cannot be classified with linear discriminants.
- we adress non-linearly separable data by introducing basis functions $phi.alt: RR^D -> RR^M$
- intuition: map data into a higher-dimensional space where it is linearly separable.
- new discriminant: $y(x) = w^sans(T) phi.alt(x)$


== Regressions

// #canvas(length: 1cm, {
//   plot.plot(size: (8, 6),
//     x-tick-step: none,
//     x-ticks: ((-calc.pi, $-pi$), (0, $0$), (calc.pi, $pi$)),
//     y-tick-step: 1,
//     {
//       plot.add(
//         style: style,
//         domain: (-calc.pi, calc.pi), calc.sin)
//     })
// })

//TODO: Neuschreiben, Logistic Regression ist "nicht wirklich" Regression sondern auch zum klassifizieren

/ Linear Regression: approximate a function $h(x)$ in RR, given by label $t_n = h(x) + epsilon$
/ Ridge Regression: Linear Regression with regularization to avoid overfitting
/ Logistic Regression: approximate a discrete function (in Linear Regression we also have only data points, but here the function is discrete)

=== Linear Regression
- we use least squares regression
- error function: $ E(w) = 1/2 sum_(i=1)^N (y(x_n; w) - t_n)^2 $
- closed-form solution: $w = (Phi^sans(T)Phi)^(-1)Phi^sans(T)t$
- basis functions: $y(x) = w^sans(T) phi.alt(x)$

=== Ridge Regression
- choosing a too high degree for (polynomial) $phi.alt$ may cause overfitting.
- introduce a regularizer $Omega$ (e.g. $Omega = 1/2||w||^2$) to the error function: 
$
E(w) = L(w) + lambda Omega(w) = 1/2 sum_(i=1)^N (y(x_n; w) - t_n)^2 + lambda/2 ||w||^2
$

note that w now penalizes large weights, which helps to avoid overfitting.  
$ w = (Phi^sans(T)Phi + lambda I)Phi^(-1) t $


=== Logistic Regression

- we model class posteriors $p(Cl_1 | x)$ directly using linear discriminants. 
- $p(Cl_1 | x) = sigma(w^sans(T)x)$, and $p(Cl_2 | x) = 1 - p(Cl_1 | x)$
- advantages over generative models:
  - less parameters, more efficient
  - nicer derivative to work with

- we can use the maximum likelihood approach, leading to the _cross entropy error_ function:
$ E(w) = - sum_(n=1)^N (t_n ln y_n + (1-t_n) ln (1-y_n)) = - ln(p(t|w)) $
- $p(t|w)$ is the product of all distances to the discriminant $y(x)$ and is defined as follows:
$ p(t|w) = product_(n=1)^N y_n^(t_n) (1-y_n)^(1-t_n) $ with $y_n = p(Cl_1 | phi.alt(x_n))$

==== Soft-Max Regression
- extension of logistic regression to K classes
- $#text("softmax(a)") = (exp(a_k))/(sum_(j=1)^K exp(a_j))$ für eine Klasse $k in underline(K)$. (Die beiden Sachen lassen sich auch verbinden :/)

==== Gradient Descent
- no closed-form solution for logistic regression
- use gradient descent to minimize $E(w)$
- iterative approach:
  - start with an initial guess for parameters $w^(0)$
  - update parameters iteratively using the gradient of the error function
  - move towards a minimum of the function by following the directiuon  of the steepest descent
  - learning rate $eta in RR^+$ controls the step size of each update (a too large $eta$ may miss the minimum, a too small $eta$ may take too long to converge)
- The update rule (also known as deltra rule, LMS rule):
$ w^(tau+1) = w^(tau) - eta nabla E(w) = w^(tau) - eta sum_(n=1)^N (y_n - t_n) phi.alt(x_n) $




== Support Vector Machines (SVM)

#figure(
  image("img/male/svm.png", height: 3cm),
  caption: [Example of a SVM],
)

- with linear discriminants, we seprated classes by a linear function $y(x) = w^sans(T)x + b$
- but there are many possible linear functions that separate the classes
- with SVMs we try to choose the best one, so that the _margin_ between the classes is maximized
- this helps with generalization to new data points (leaving maximal safety room for future noisy data)

This is done as follows:
- for _every_ data point $x_n$ with label $t_n in {-1, 1}$ we want $t_n y(x_n) >= 1$ 
- we choose w so that distance between the decision boundary and the closest data point is $1/(||w||)$
- maximize margin by minimizing $||w||^2$ (easier than $||w||$)

==== The SVM Objective
- find w, b  to minimize $1/2 ||w||^2$ such that $t_n (w^sans(T)x_n + b) >= 1$ for all $n in N$
- the constraint ensures all data points are correctly classified
- this is a constarined optimization  problem  with $ K(x) = 1/2 ||w||^2  #text("and") f_n (x) = t_n (w^sans(T)x_n + b) - 1 >= 0 $
- the problem can be solved with Lagrange multipliers $a_n$ and Karush-Kuhn-Tucker conditions 
- Karush-Kuhn-Tucker conditions:
$
       lambda & >= 0 \
         f(x) & >= 0 \
  lambda f(x) & = 0
$



==== Primal Form
$ L(w,b,a)=1/2 ||w||^2 - sum_(n=1)^N a_n [t_n (w^sans(T)x_n + b) -1] $

KKT Conditions:
$
                            a_n & >= 0 \
       t_n (w^sans(T)x_n + b) -1 & >= 0 \
  a_n [t_n (w^sans(T)x_n + b) -1] & = 0
$

We want to minimize L(w, b, a) w.r.t. w and b. by setting the derivatives to zero, we get:
$
       partial/(partial w) L(w,b,a) & = w - sum_(n=1)^N a_n t_n x_n = 0 \
       partial/(partial b) L(w,b,a) & = sum_(n=1)^N a_n t_n = 0
$
- This gives us a weight to how much each data point influences the decision boundary. 
- Points with $a_n > 0$ (points on the margin) are called _support vectors_ and influence the decision boundary. 
- Points with $a_n = 0$ (on the either side) do not influence the decision boundary, making SVM robust to "too correct" data points- 

We already have w from above, but we still need to find b: 
$
  b = 1/N_S sum_(n in S) (t_n - sum_(m in S) a_m t_m (x^sans(T) x_n))
$
where S is the set of support vectors. Now we need to find the Lagrange multipliers $a_n$ by solving the _quadratic programming_ problem (cubic complexity, use tools to compute).


==== Dual Form
- not dependet on the diemensionality D of the data, only on the number of data points N. This _may seem_ worse, but in practice N is often much smaller than D.

$ L_d(a)=sum_(n=1)^N a_n - 1/2 sum_(n=1)^N sum_(m=1)^N a_n a_m t_n t_m (x^sans(T) x_n) $

with conditions:
$
                  a_n & >= 0 quad forall n in underline(N) \
  sum_(n=1)^N a_n t_n & = 0
$

==== Soft-Margin SVMs
- Data is not always linearly separab1le.
- our current SVM formulation would fail in this case.
- Now we allow some data points to be on the wrong side of the margin using _slack variables_ $xi_n = |t_n - y(x_n)|$ for all training points.
- the slack variable is a linear penalty that corresponds to the distance we need to move a misclassified point to the correct side of the margin.
- if $xi_n = 0$, the point is correctly classified, otherwise it incurs a penalty up to 1.
- New constraint: $t_n (w^sans(T)x_n + b) >= 1 - xi_n$ with $xi_n >= 0$ for all n in underline(N)


This leads to the new primal objective:
- minimize $1/2 ||w||^2 + C sum_(n=1)^N xi_n$ such that $t_n (w^sans(T)x_n + b) >= 1 - xi_n$ and $xi_n >= 0$.
- C is a hyperparameter that controls the trade-off between maximizing the margin and minimizing the classification error.


The new dual formulation is exactly the same, with almost the same KKT conditions. The only difference is that now $0 <= a_n <= C$.


==== Non-Linear SVMs
- For non-linearly separable data, we can use kernel functions to map the data into a higher-dimensional space where it becomes linearly separable. (using a basis function $phi.alt$)
- The kernel trick allows us to compute dot products in the high-dimensional space without explicitly mapping the data 
- Note that this works because we never need to compute $phi(x)$ explicitly, but only in an inner product form.
- A kernel function $k: RR^D times RR^D -> RR, (x_1, x_2) mapsto phi.alt(x_1)^sans(T) phi.alt(x_2)$ 
- The dual formulation of the SVM now becomes:
$ L_d(a)=sum_(n=1)^N a_n - 1/2 sum_(n=1)^N sum_(m=1)^N a_n a_m t_n t_m k(x_n, x_m) $
- classify new data points using:
$ y(x) = sum_(n=1)^N a_n t_n k(x_n, x) + b $


== Error Function Analysis

#stack(
  dir: ltr,
  spacing: 3em,
  {
    figure(
      image("img/male/error-functions.png", height: 10em),
      caption: [Error Funktionsanalyse],
    )
  },

  {
    figure(
      image("img/male/error-functions-2.png", height: 10em),
      caption: [Weitere Error Funktionsanalyse],
    )
  },
)

There are 2 error contribbution plots above:
- left of y-axis: incorrect classifications
- right of y-axis: correct classifications, the more positive the better
- we plot the error contribution on the y axis

- the optimal error function is the step function, but it has no gradient, so gradient descent is not possible
- the squared error increases too fast for negative values, and also penalizes "too correct" points (far away from the boundary)
- hinge loss is not differentiable at x=1, but has _sparcity_ for x>1 (only points on the boundary matter)
- cross-entropy error increases linearly for negative values (good for gradient descent) and has no positive penalty


== Neural Networks (NN)

A _perceptron_ is a node in a neural network (NN). They are basically generalized linear discriminants. 

_Single layer perceptron_:
- Input: $x in RR^D$ hand-designed features, with $w in RR^D$ weights and bias $w_0 in RR$.
- Output at node k: 
  -  linear: $y_k (x) = sum_(i=0)^D w_(k i) phi.alt(x_i))$
  -  logistic: $y_k (x) = sigma(sum_(i=0)^D w_(k i) phi.alt(x_i))$
- Goal: learn weights and bias from data.

_Multi-Layer Perceptron (MLP)_:
- weights are now a matrix W
- each layer computes a matrix multiplication and applies an elementwise activation function (sigmoid etc.):
$ z(x) = g^((1))(W^((1)) x) $ 
$ y(x) = g^((2))(W^((2)) z(x)) $
$ ... $
$ y_k (x) = g^((2)) ( sum_(i=0)^M w_(k i)^((2)) g^((1)) ( sum_(j=0)^D w_(i j)^((1)) x_j ) ) $


- this makes earlier layers also learnable. 
- usually, each layer adds a bias term. 
- activation _needs_ to be non-linear!!!
- much more powerful than single-layer perceptrons (universal approximators).

==== Training NNs

#figure(
  image("img/male/nn-training-pipeline.jpeg", height: 8em),
  caption: [Neural Network Training Pipeline],
)

1. Error function:
  - combination of _loss function_ $L(t, y(x))$ and _regularizer_ $Omega (x)$
  - typical loss functions: squared error, cross-entropy error, hinge loss
  - the loss function we use determines the type of problem we are solving (regression, classification, etc.) 
  - typical regularizers: L2 norm ($1/2 ||w||_2^2$), L1 norm ($1/2 ||w||_1$)

2. Backpropagation:
  - compute the gradient of the error function w.r.t. all weights by propagating the error backwards through the network
  - naive approach: $(partial z) / (partial x) = sum_(i=1)^k (partial z) / (partial y_i) (partial y_i) / (partial x)$
  - efficient approach: compute the gradients layer by layer, reusing intermediate results
  - implemeted via _automatic differentition_ (convert the network into a computtaion graph and apply backpropagation on it)

#figure(
  image("img/male/comp-graph.png", height: 10em),
  caption: [Computation Graph of logistic regression with cross-entropy loss],
)

#figure(
  image("img/male/simple-comp-graph.png", height: 12em),
  caption: [Simplified Computation Graph],
)

#pagebreak()

= Empirical Analysis and Performance Optimization


== Data Quality and Preprocessing
- Goal: increase data quality and modify the data to suit the analysis question and applied techniques, "Garbage in, garbage out"

- Quality aspects: accuracy, completeness, consitency, timeliness

=== Missing Values 
For example, an image missing a few pixel, what do we do?
- ignore:
  - discard the whole feature (colomn in the data table)
  - discard the instance (row in the data table)
  - ignore the instance only for features where the value is missing
- create: 
  - mean/median/mode of the whole feature
  - mean/median/mode o of all instances belonging to the same class
  - use a predictor model (decison trees, regression, NNs, SVMs, ...)

=== Outliers 
For example, a data point far away from it's class' cluster is considered an outlier. We can treat it as a missing value and ignore or replace it (depending on the use case).

=== Preprocessing

- / Transformation: change the data to the right data type
- / Normalization: adjust the influence of features:
  - Min-max normalization: TODO
  - Standard score (Z-score) normalization: TODO
-/ Reduction: make the data smaller for analysis

== Responisble Data Science
Four key concerns: Fairness, Accuracy, Confidentiality, and Transparency ("FACT"!)

This chapter mainly contains metrics to compute confidentiality and fairness of metrics we have already seen before. 

=== Confidentiality
"How to answer questions without revealing secrets?"

To achieve confidentiality, we anonimize the data table and build equivalence classes ECs (instances in the same value range, like age: 20-25, 30-35, 35+ etc.). We can measure it using 3 metrics:

- / k-Anonymity: each equivalence class contains at least k instances
- / Entropy l-diversity: a data table is l-diverse if for every EC: $H(E  C) >= log_2(l)$
- / t-Closeness: EC has t-closeness if the distance between the distribution DE of a sensitive feature in this class and the distribution DT of it in the whole table is no more than a threshold t: $D i s t a n c e("DE", "DT") <= t$

=== Fairness
"How to avoid unfair conclusions even if they are true?". 

We have 2 metrics: 

- / Discrimiation: $d i s c_cal(D)(cal(B)) = 4 dot |support(cal(B) union cal(D)) - support(cal(D)) dot support(cal(B))|$, where:
  - $cal(B)$ is the  possible outcome, i.e., the target feature(s) having particular vales (e.g.: $cal(B) = {"outcome" = "expected"}$) 
  - $cal(D)$ is the potentially discriminating itemset characterizing the properties of instances we do not want to have any effect (e.g.: $cal(D) = {"gender" = "male"}$)
  - This yields a value between 1 (maximal discrimination) and 0 (no discrimination)
  - Propertiy: $d i s c_cal(D)(cal(B))  = d i s c_cal(D)(cal(overline(B))) = d i s c_cal(overline(D))(cal(B)) = d i s c_cal(overline(D))(cal(overline(B))) $

- / Information Gain in Sensitivity: Decision tree metric $I G S (d) = H(b) - H^d_W (b)$, where:
  - d is the split feature, b is the sensitive feature
  - We want the lowest IGS possible, and the highets IGC (classical IG) possible

== Evaluation 
#let TP = math.op("TP")
#let TN = math.op("TN")
#let FP = math.op("FP")
#let FN = math.op("FN")

Main questions: how good is the ML model, how good can it be?

#figure(
  image("img/male/eval-matrix.png", height: 12em),
  caption: [Confusion Matrix],
) <confusion-matrix>

We have 4 cases we can classify our prediction as:
- True positive $TP$ (correctly labeling a positive label)
- True negative $TN$ (correctly labeling a negative label)
- False positive $FP$ (incorrectly labeling a positive label)
- False negative $FN$ (incorrectly labeling a negative label)

We have multiple measures to evaluate the quality of our prediction model:
- / Accuracy: $(TP+TN)/#highlight(fill: yellow)[(TP+TN+FP+FN)]$
- / Misclassification Rate: $(FP + FN)/#highlight(fill: yellow)[(TP+TN+FP+FN)]$

- / #highlight(fill: blue)[True Positiv] Rate (Recall): $TP/#highlight(fill: blue)[(TP+FN)]$
- / #highlight(fill: blue)[False Negative] Rate: $FN/#highlight(fill: blue)[(TP+FN)]$

- / #highlight(fill: green)[True Negative] Rate: $TN/#highlight(fill: green)[(TN+FP)]$
- / #highlight(fill: green)[False Positive] Rate: $FP/#highlight(fill: green)[(TN+FP)]$
- / Precision: $TP/(TP+FP)$ (if we make a positive prediction, how often is it correct?)
- / $F_1$ Score: $2 dot ("precision" dot "recall")/("precision" + "recal") $ (geometric mean between _recall_ and _precision_)

=== Training and Test Data
- We split our data set into: training and testing. We train the predictivve model with the training set and evaluate it's performance with the testing set.
- The training set is also split into: training and validation. While training, we need to validate the choice of hyperparameters chosen to finally optimize the performance.
- Rule of thumb: 50% training set, 20% validation set, 30% testing set (can vary +-10%)

=== How do we deal with small data sets? 
We train the model k times and evaluate it k times, while varying our choice of training and test sets throughout the iterations:

/ k-Fold Cross Validation: 
- split data into equally sized chunks ($N/k$)
- pick one chunk as the test set, the rest is training sets
- repeat until each partition becomes the test set once (k times)
- quality score is the average over the k folds

/ Jackknifing: k-Fold with $k=N$ (meaning test set is 1 single instance)
/ Bootstrapping: randomly pick $m in NN$ test instances, repeat k times 


== AutoML

=== Hyperparameter Optimization

Es gibt bei uns grob 4 Methoden:
- Random Search
- Grid Search
- Bayesian Optimization
- Multi Fidelity Bandit

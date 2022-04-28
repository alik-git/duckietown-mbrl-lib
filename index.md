---
title: Duckietown MBRL-Lib
---

<!-- MathJax -->
<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

___

**Authors:** [Ali Kuwajerwala](https://alik-git.github.io/), [Paul Crouther](https://www.linkedin.com/in/paul-crouther-47221b52/) <br /> 
**Affiliation:** [Université de Montréal](https://diro.umontreal.ca/accueil/), [Mila](https://mila.quebec/en/) <br />
**Date Published:** April 28, 2022

___




**Abstract**

Model-based reinforcement learning (MBRL) algorithms have various sub-components that each need to be carefully selected and tuned, which makes it difficult to quickly apply existing models/approaches to new tasks. In this work, we aim to integrate the Dreamer algoithm into an existing popular MBRL toolbox, and tune the Dreamer-v1 and PlaNet algorithms to the Gym-Duckietown environment. We also provide trained models and code to to use as a baseline for further development. Additionally, we propose an improved reward function for RL training in Gym-Duckietown, with code to allow easy analysis and evaluation of RL models and reward functions.

states($s_t$), actions(\$a_t\$) and observations($o_t$)

- [Introduction](#introduction)
- [Related Work](#related-work)
  - [Model Based vs Model Free RL](#model-based-vs-model-free-rl)
  - [MBRL-Lib](#mbrl-lib)
  - [Gym-Duckietown](#gym-duckietown)
  - [Dream to Control (Dreamer)](#dream-to-control-dreamer)
- [Background](#background)
  - [Model-Based Reinforcement Learning](#model-based-reinforcement-learning)
  - [PlaNet for Gym-Duckietown](#planet-for-gym-duckietown)
  - [Model Based vs Model Free RL](#model-based-vs-model-free-rl-1)
- [Project Method](#project-method)
  - [Dreamer](#dreamer)
  - [Dreamer training](#dreamer-training)
  - [Dreamer evaluation](#dreamer-evaluation)
- [New skills we learned](#new-skills-we-learned)
  - [What we learned with Dreamer](#what-we-learned-with-dreamer)
  - [What we learned about logging and organization](#what-we-learned-about-logging-and-organization)
- [Experiments and Analysis](#experiments-and-analysis)
- [Video Results](#video-results)
- [Conclusions](#conclusions)
- [Work Division](#work-division)


## Introduction 


    1 [4 pts] Project Introduction (2 paragraphs, with a figure)
    Tell your readers why this project is interesting. What we can learn. Why it is an important area for robot learning.

This project is interesting because we would like to have robots perform difficult tasks within large environments with complex dynamics. MBRL is still very difficult for such tasks since the learned world model must accurately represent and make predictions about the environment. Model free (MF) approaches can perform better on such tasks, but they typically require an amount of interaction with the environment that is very inefficient when compared to MB approaches. 

Therefore, improving MBRL approaches to outperform MF approaches for tough environments would make it easier to develop robot solutions for the tasks we care about, since the barrier to entry posed by the inefficient sample complexity would be eliminated. Additionally, the best learned world models could be reused for a variety of tasks. 

In this work we attempt to use two MBRL algorithms, Dreamer-v1 and PlaNet, to learn a world model for the Gym-Duckietown environment, which we then use to train a lane-following policy. The Gym-Duckietown environment is significantly larger and more complex than environments included in the aforementioned algorithms' respective papers. We also provide the code we used, where we place a strong emphasis on documentation and modularity, so that more MBRL approaches can be applied to Gym-Duckietown, where our Dreamer-v1 and PlaNet can serve as comparative baselines. This helps us move toward our goal of getting robots to perform the kinds of difficult tasks we care about.

##  Related Work

    2 [2 pts] What is the related work? Provide references:(1-2 paragraph(s))
    Give a description of the most related work. How is your work similar or different from the most related work?


<!-- ### RL, MBRL, PlaNet, what we can improve on PlaNet
*Dreamer is different because it uses PlaNet as a world model, and gets value estimates with respect to the world model, and then takes actions with respect to those estimates -->

### Model Based vs Model Free RL

Reinforcement learning is an active field of research in autonomous vehicles. Model-based approaches are used less frequently than model-free approaches since model-free approaches have had better performance in the past and as a result it is easier to find existing implementations. 

The largest advantage that model based approaches offer is their superior sample complexity. That is, model based approaches can use orders of magnitude less data or interaction with the external environment compared to model-free methods. This is because model based methods can train the policy on the internal model of the environment as opposed to the external environment itself. Additionally, another advantage that model-based methods have is that the learned model of the environment can be task agnostic, meaning that it can be used for any task that requires predicting the state of the environment in the future.


### MBRL-Lib

There are many available open source implementations of popular \mf approaches \cite{baselines, pytorchrl} and \mb approaches \cite{bacon}. During our review, we found the MBRL-Lib toolbox \cite{MBRL} to be most useful. 

MBRL-Lib contains implementations of various popular \mb approaches, but more importantly it is designed to be modular and compatible with a wide array of environments; and makes heavy use of configuration files to minimize the amount of new code needed to tweak an existing approach. Integrating the Gym-Duckietown environment to work with this toolkit would allow users to easily experiment with and evaluate a variety of model based RL approaches in the Duckietown simulation.

### Gym-Duckietown


Gym-Duckietown, a self-driving car simulator for the Duckietown universe already built as an OpenAI Gym environment is the ideal  candidate to make available for use with the model-based approaches provided by MBRL-Lib along with Dreamer.

Gym-Duckietown is different from most environments previously used by model-based approaches. It has significantly more complex dynamics than the standard \og environments. Consider for example that the Cheetah environment (See Fig. \ref{fig:fig1a}) only consists of one (albeit complex) object in a plain background with the camera always tracking it. Compared to \gd, where the camera is fixed on the car which moves through the scene, drastically changing the objects found in different observations.
Our results indicate that while MBRL methods have the potential to perform well in \gd, they need to be carefully tuned and modified to achieve results comparable to those of the current baselines.

### Dream to Control (Dreamer)

To observe how Duckietown scales with different and more performant MBRL algorithms and to facilitate learning, one of the goals in this project is to develop an implementation of Dreamer \citep{Dreamer}, which does not exist in MBRL-Lib. Similar to PlaNet, Dreamer uses a recurrent state space model (RSSM) to represent the underlying MDP with partial observability by using PlaNet as a world model. Where it differs from PlaNet is in the model fitting section and the lack of a planning section. With Dreamer, the model fitting part is broken into a dynamics learning section and a behavior learning section which does rollouts of imagined trajectories, and finally updates parameters for the action model, and value model using gradients of value estimates of imagined states for the learning objective. Since these trajectories are imagined, the authors utilize reparameterization for continuous actions and states. Lastly, Dreamer computes the state from history during the environment interaction step, with noise added to the action for exploration.

In the paper, the authors run Dreamer on the DeepMind control suite, similar to PlaNet. However, since the original implementation is in TensorFlow, it will need to be re-implemented in PyTorch for direct comparison to other algorithms in MBRL-Lib.

## Background

    3 [2 pts] What background/math framework have you
    used? Provide references: (2-3 paragraph(s) + some math)
    Describe what methods you are building your work on. 
    Are you using RL, reset-free, hardware, learning from images? 
    You want to provide enough information for the average student in the class to understand the background.

### Model-Based Reinforcement Learning

To setup the reinforcement learning problem from a model-based reinforcement learning (MBRL) perspective, we adhere to the Markov decision process formulation \citep{bellman1957markovian}, where we use state $s \in \mathcal{S}$ and actions $a \in \mathcal{A}$ with reward function  $r(s,a)$ and the dynamics or transition function $f_\theta$, such that $s_{t+1} = f_{\theta}(s_t, a_t)$ for deterministic transitions, and stochastic transitions are given by the conditional $f_\theta(s_{t+1}\mid s_t, a_t) = \mathbb{P}(s_{t+1}\mid s_t, a_t, ; \theta)$ and learning the forward dynamics is akin to doing a fitting of approximation $\hat{f}$ to the real dynamics $f$ given real data from the system.\\

### PlaNet for Gym-Duckietown

Of the important contributions of PlaNet \citep{PlaNet}, one of them is the recurrent state space model (RSSM). The RSSM has both stochastic and deterministic components and it was shown in PlaNet to greatly improve results compared to purely stochastic or deterministic models on complicated task.
To bring the input images to the latent space, we need an encoder. Since we are using images, a convolution neural net is perfect for the task.\\
No-policy is actually trained since the planning algorithm use only the models to choose the next best action.\\

Since the models are using stochastic decisions, the training is using a variational bound to optimise its parameters. It alternatively optimises the encoder model and the dynamics model by gradient ascent over the following variational bound:

$$\ln{p}(o_{1:T}  \mid a_{1:T}) = \ln \int \prod_t p(s_t\mid s_{t-1},a_{t-1})p(o_t\mid s_t)ds_{1:T} \\ \geq \sum_{t=1}^{T}  \bigg( \mathbb{E}_{q(s_t\mid o_{\leq t},a_{\leq t})}[\ln{p(o_t\mid s_t)}])  - \\
\mathbb{E}_{q(s_{t-1}\mid o_{\leq t-1},a_{\leq t-1})}[KL[q(s_{t}\mid o_{\leq t},a_{\leq t})\mid \mid  p(s_t\mid s_{t-1},a_{t-1})]] \bigg)$$

The PlaNet models follow a Partially Observable Markov Decision Process(POMDP). It is built on a finite sets of: states($s_t$), actions($a_t$) and observations($o_t$).

* Dynamics model : $s_t \sim p(s_t \mid s_{t-1},a_{t-1})$
* Encoder : $s_t \sim q(s_t \mid o_{\leq t},a_{\leq t})$
* Decoder : $o_t \sim p(o_t \mid s_{t},a_{t-1})$
* Reward : $r_t \sim p(r_t \mid s_t)$
* $\gamma$ the discount factor $\gamma \in [0,1]$

The rest of the details are outlined for the RSSM representation in comparsion to deterministic and stochastic models is shown in figure \ref{fig:planet_rssm}.

###  Model Based vs Model Free RL

The following section is the answer for both question 2 and question 3. The majority of this answer was taken directly from Ali's previous project since the related work has not changed. We have added a section for Dreamer.


Reinforcement learning is an active field of research in autonomous vehicles. Model-based approaches are used less frequently than model-free approaches since model-free approaches have had better performance in the past and as a result it is easier to find existing implementations. 

The largest advantage that model based approaches offer is their superior sample complexity. That is, model based approaches can use orders of magnitude less data or interaction with the external environment compared to model-free methods. This is because model based methods can train the policy on the internal model of the environment as opposed to the external environment itself. Also, model-free methods implicitly learn a `model of the environment' in some way eventually in order to predict the long-term reward of an action, they just do so inefficiently. Additionally, another advantage that model-based methods have is that the learned model of the environment can be task agnostic, meaning that it can be used for any task that requires predicting the state of the environment in the future.

##  Project Method

    4 [6 pts] Project Method, How will the method work (1-2 pages + figure(s) + math + algorithm)
    Describe your method. 
    Again, You want to provide enough information for the average student in the class to understand how your method works. 
    Make sure to include figures, math, or algorithms to help people understand.

### Dreamer

Given that the MBRL-lib does not include a Dreamer implementation, but rather the PlaNet model, it makes sense to reuse the recurrent state model from the PlaNet implementation as a world model for Dreamer. However, there are structural components that are missing. For example, in departure from PlaNet, rather than a CEM for the best action sequence under the model for planning, Dreamer uses a dense action network parameterized by $\phi$ and the dense value network parameterized by $\psi$. For the action model with imagined actions, the authors use a tanh-transformed Gaussian \citep{SAC} output for the action network, which provides a deterministically-dependent mean and variance of the state through a reparameterization \citep{kingma2013auto} \citep{rezende2014stochastic} of the stochastic node, adding noise $\epsilon$ back in afterwards (to be inferred).

$$a_\tau = \tanh(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau) \epsilon), \qquad \epsilon \sim \mathcal{N}(0, \mathbb{I})$$

This formalizes the deterministic output of the action network returns a mean $\mu_\phi(s_\tau)$ and we learn the variance of the noise $\sigma_\phi(s_\tau)$ with this reparameterization, inferring from our normal noise $\epsilon$, to represent our stochastic model.

Then the value network consists of imagined value estimates $V_R(s_\tau) = \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} r_n))$ which is the sum of rewards until the end of a horizon, then using values $v_\psi(s_\tau)$ then computes $V^{k}_N(s_\tau) = \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{h = \min(\tau+k, t+H)-1} \gamma^{n-\tau}r_n + \gamma^{h-\tau} v_\psi(s_h)))$ as a estimate of rewards beyond $k$ steps with the learned value model, and $V_\lambda(s_\tau)$ which is a exponentially weighted average of $V^{k}_N(s_\tau)$ at different values of k, shown in the following. 

$$V_\lambda(s_\tau) = (1 - \lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V_{N}^{n}(s_\tau) + \lambda^{H-1}V_{N}^{H}(s_\tau)$$

This helps Dreamer do better with longer-term predictions of the world, over shortsightedness with other types of dynamics models for behavior learning. Since Dreamer disconnects the planning and action by training an actor and value network and uses analytic gradients and reparameterization, it is more efficient than PlaNet which searches the best actions among many predictions for different action sequences. This motivates the implementation of Dreamer to compare to PlaNet with potential performance improvements with a similar number of environment steps. The policy is trained via using the analytical gradient $\nabla_\phi \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} V_\lambda(s_\tau))$ from stochastic backpropagation, which in this case becomes a deterministic node where the action is returned shown in \ref{eq:1}, with the value network being updated with the gradient  $\nabla_\psi \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} \frac{1}{2}\\mid v_\psi(s_\tau) - V_\lambda(s_\tau))\\mid ^2$ after imagined value estimates are computed. All of this happens in the update steps for behavior and dynamics learning. Finally, in an environment interaction time step, the agent gets states from its history and returns actions from the action network, and value model estimates the imagined rewards that the action model gets in each state. These are trained cooperatively in a policy iteration fashion.

### Dreamer training

For training the first version of the Dreamer prototype, we used Cheetah the environment to compare directly to the in library PlaNet implementation. with action noise of 1.0 and 0.3 like the original paper. The model, actor, and critic losses are logged from their respective networks. The model loss contains the reconstruction loss and represents the PlaNet world or dynamics model, represented by an encoder and decoder (a va). The actor loss is the loss from the actor network, which is a 

### Dreamer evaluation 

## New skills we learned

    5 [2 pts] What new skills have you(s) learned from this project?
    List some of the technical skills you are learning from studying this method.

### What we learned with Dreamer

- Dreamer fundamentally different than other MBRL algorithms in the sense that there is also a policy being learned as opposed to just a world model.



- MBRL-Lib tries to think of MBRL approaches as models only, and uses a "universal" outer loop to train the models, with the details of each model being inside a "train" function for that model. Dreamer is not very well suited in practice to this approach and makes it the optimization of the 3 networks difficult. This made us learn about the general structure of RL algorithms and MBRL algorithms.

- Goal in mind was to do dreamer implementation that fits well with MBRL-Lib, potentially to submit a pull request to the library. But we decided to first get a prototype implementation done as a proof of concept that Dreamer can be trained using the env and training loop of the library.

### What we learned about logging and organization

- logging is tough


## Experiments and Analysis

    6 [8 pts] Experiments and Analysis**
    In this section
    1. Describe what experiment(s) you are going to run and why? How do these show you have met your learning goals?
    2. What do you think the results of these experiments will be?
    3. Sketch out the figures that you will later generate from your work. Spend a few
    minutes drawing them out in GIMP or photoshop. Why will these be enough evidence
    for learning? Is anything missing?

    Keep in mind these experiments are for this course project. What is expected is that you
    should provide evidence that your method works and it has been coded up well. Provide
    evidence of this via your data and learning graphs. However, this should not be restricted
    to learning graphs.


## Video Results 

    7 [2 pts] Video Results
    Include a link to a video of the method so far. 
    This is not intended to be a final performance video. 
    Provide something to help me give feedback on what I would do to improve this method or debug issues.


## Conclusions

    8 [4 pts] Conclusions
    What have your results indicated?
    What have you learned? 
    What would you do differently next time? 
    Reflect on the scope of your project, was it too much? Why?


## Work Division

    Provide a description on what each group member is working on as part of the project. 
    I recommend each student work on most of the parts of the project so everyone learns about the content.
    Student Name: Did x, y, and z.
    Student Name: Did x, q, and r.
    Student Name: Did q, y, and r.


You can use the [editor on GitHub](https://github.com/alik-git/duckietown-mbrl-lib/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.



Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block



- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

Jekyll Themes 

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/alik-git/duckietown-mbrl-lib/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

Display equation: $$equation$$

$ y = f(x)$
- $x + y$
- $x - y$
- $x \times y$ 
- $x \div y$
- $\dfrac{x}{y}$
- $\sqrt{x}$


- $\pi \approx 3.14159$
- $\pm \, 0.2$
- $\dfrac{0}{1} \neq \infty$
- $0 < x < 1$
- $0 \leq x \leq 1$
- $x \geq 10$
- $\forall \, x \in (1,2)$
- $\exists \, x \notin [0,1]$
- $A \subset B$
- $A \subseteq B$
- $A \cup B$
- $A \cap B$
- $X \implies Y$
- $X \impliedby Y$
- $a \to b$
- $a \longrightarrow b$
- $a \Rightarrow b$
- $a \Longrightarrow b$
- $a \propto b$

$$\mathbb{N} = \{ a \in \mathbb{Z} : a > 0 \}$$
$$\forall \; x \in X \quad \exists \; y \leq \epsilon$$

$$\color{blue}{X \sim Normal \; (\mu,\sigma^2)}$$
$$P \left( A=2 \, \middle\mid  \, \dfrac{A^2}{B}>4 \right)$$
$$f(x) = x^2 - x^\frac{1}{\pi}$$
$$f(X,n) = X_n + X_{n-1}$$
$$f(x) = \sqrt[3]{2x} + \sqrt{x-2}$$

https://ashki23.github.io/markdown-latex.html

```latex

















%%%%%%%% ICML 2018 EXAMPLE LATEX SUBMISSION FILE %%%%%%%%%%%%%%%%%

\documentclass{article}

% Recommended, but optional, packages for figures and better typesetting:
% \usepackage[demo]{graphicx}
% \usepackage{subcaption}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
% \usepackage{subfigure}
% \usepackage{subfig}
\usepackage{booktabs} % for professional tables
\usepackage{amsmath, amssymb}
\usepackage{lipsum}
\usepackage{listings}
\usepackage{minted}
% \usepackage{hyperref}
% \usepackage{autoreff}
\setlength{\belowcaptionskip}{0pt}

\usepackage{xcolor}
%New colors defined below
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

%Code listing style named "mystyle"
\lstdefinestyle{mystyle}{
  backgroundcolor=\color{backcolour}, commentstyle=\color{codegreen},
  keywordstyle=\color{magenta},
  numberstyle=\tiny\color{codegray},
  stringstyle=\color{codepurple},
  basicstyle=\ttfamily\footnotesize,
  breakatwhitespace=false,         
  breaklines=true,                 
  captionpos=b,                    
  keepspaces=true,                 
  numbers=left,                    
  numbersep=5pt,                  
  showspaces=false,                
  showstringspaces=false,
  showtabs=false,                  
  tabsize=2
}

%"mystyle" code listing set
\lstset{style=mystyle}

\graphicspath{ {figures/} }
% \hyphenpenalty=
\usepackage{stackengine}
\def\delequal{\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptstyle\Delta}}}}

\usepackage{svg}
% \usepackage[nohyperref]{icml2018}
% hyperref makes hyperlinks in the resulting PDF.
% If your build breaks (sometimes temporarily if a hyperlink spans a page)
% please comment out the following usepackage line and replace
% \usepackage{icml2018} with \usepackage[nohyperref]{icml2018} above.
\usepackage{hyperref}

% Attempt to make hyperref and algorithmic work together better:
\newcommand{\theHalgorithm}{\arabic{algorithm}}

\newcommand{\mb}{model-based }
\newcommand{\mf}{model-free }
\newcommand{\gd}{Gym-Duckietown}
\newcommand{\ml}{MBRL-Lib }
\newcommand{\og}{OpenAI Gym }
\newcommand{\hc}{HalfCheetah-v2}
\newcommand{\ls}{\lstinline}
% Use the following line for the initial blind version submitted for review:
% \usepackage{icml2018_ift6269}

% If accepted, instead use the following line for the camera-ready submission:
\usepackage[accepted]{icml2018_ift6269}
% SLJ: -> use this for your IFT 6269 project report!

% \definecolor{battleshipgrey}{rgb}{0.52, 0.52, 0.51}
\newcommand{\crr}{\color{red}}
\newcommand{\cbb}{\color{blue}}
\newcommand{\cgg}{\color{gray}}

% The \icmltitle you define below is probably too long as a header.
% Therefore, a short form for the running title is supplied here:
\icmltitlerunning{Model Based RL in Gym-Duckietown}

\begin{document}

\twocolumn[
\icmltitle{IFT6163 Final Report - Model Based RL in Gym-Duckietown} % Dynamic Duckies on a Diet

% It is OKAY to include author information, even for blind
% submissions: the style file will automatically remove it for you
% unless you've provided the [accepted] option to the icml2018
% package.

% List of affiliations: The first argument should be a (short)
% identifier you will use later to specify author affiliations
% Academic affiliations should list Department, University, City, Region, Country
% Industry affiliations should list Company, City, Region, Country

% You can specify symbols, otherwise they are numbered in order.
% Ideally, you should not use this facility. Affiliations will be numbered
% in order of appearance and this is the preferred way.

\icmlsetsymbol{equal}{*}

\begin{icmlauthorlist}
\icmlauthor{Alihusein Kuwajerwala}{equal,udem,mila}
\icmlauthor{Paul Crouther}{equal,udem,mila}
% \icmlauthor{Brahim Ben Malek}{equal,udem}
\end{icmlauthorlist}



\icmlaffiliation{udem}{Université de Montréal, Montréal, Canada}
\icmlaffiliation{mila}{Mila - Quebec AI Institute, Montreal, Canada}

\icmlcorrespondingauthor{Alihusein Kuwajerwala}{alihusein.kuwajerwala@umontreal.ca}


% You may provide any keywords that you
% find helpful for describing your paper; these are used to populate
% the "keywords" metadata in the PDF but will not be shown in the document
\icmlkeywords{Machine Learning, ICML}

\vskip 0.3in
]

% this must go after the closing bracket ] following \twocolumn[ ...

% This command actually creates the footnote in the first column
% listing the affiliations and the copyright notice.
% The command takes one argument, which is text to display at the start of the footnote.
% The \icmlEqualContribution command is standard text for equal contribution.
% Remove it (just {}) if you do not need this facility.

% \printAffiliationsAndNotice{}  % leave blank if no need to mention equal contribution
\printAffiliationsAndNotice{\icmlEqualContribution} % otherwise use the standard text.

\begin{abstract}

% In this work, we investigate a model based reinforcement learning algorithm proposed in the `PlaNet' paper \cite{PlaNet} that attempts to learn a dynamics model in the latent space for the environment. We outline and compare the available open source implementations of this algorithm for research. For the bulk of our experiments we used MBRL-Lib \cite{MBRL}, a toolbox for deploying various model based rl methods, that already contains an implementation of the PlaNet algorithm. We implemented the required boilerplate code to use the Duckietown gym environment with MBRL, and compare the PlaNet algorithms performance on the Duckietown environment and the Cheetah Run environment.

% \\  Model-based reinforcement learning (MBRL) offers a framework for data-efficient learning of agents for various tasks. 
% Despite theoretical evidence suggesting that model-based reinforcement learning (MBRL) has superior sample complexity compared to model-free methods, MBRL approaches have struggled to match their performance. This is likely because most MBRL algorithms have many complex sub-components that each need to be carefully selected and tuned, making them harder to deploy in most practical settings. 

% Model-based reinforcement learning (MBRL) algorithms have many complex sub-components that each need to be carefully selected and tuned, which may explain why MBRL approaches often struggle to match the performance of model-free methods on various tasks. In this work, we aim to make it easier to evaluate and improve MBRL approaches by integrating the Gym-Duckietown environment \cite{dtown} into MBRL-Lib \cite{MBRL}, an open source library of popular MBRL algorithms. Our goal is to enable MBRL approaches to be easily deployed in the Duckietown environment without significant software overhead. Additionally, we apply the MBRL approach from `PlaNet' \cite{PlaNet}, attempting to learn a dynamics model for the Duckietown simulation, and compare its performance on the Duckietown and  `Cheetah Run' environments. The code can be found at: \\ \lstinline{https://github.com/alik-git/mbrl-lib}
Abstract is work in progress:
Model-based reinforcement learning (MBRL) algorithms have various sub-components that each need to be carefully selected and tuned, which makes it difficult to quickly apply existing models/approaches to new tasks. In this work, we aim to tune the Dreamer-v1 and PlaNet MBRL algorithms to the Gym-Duckietown environment \cite{dtown}, and provide trained models and code to to use as a baseline for further development. Additionally, we propose an improved reward function for RL training in \gd, and code to allow easy analysis and evaluation of RL models and reward functions.

\textbf{ IMPORTANT NOTE: { \crr{  Professor's questions are in red. } }  Answers are in black. { \cgg { Parts of this report that were directly taken from Ali's older project are in gray. } } }

% Model-based reinforcement learning (MBRL) algorithms have various sub-components that each need to be carefully selected and tuned, which may explain why MBRL approaches often struggle to match the performance of model-free methods on various tasks. In this work, we aim to make it easier to evaluate and improve MBRL approaches by integrating the Gym-Duckietown environment \cite{dtown} into MBRL-Lib \cite{MBRL}, an open source library of popular MBRL algorithms. Our goal is to enable MBRL approaches to be easily deployed in the Duckietown environment without significant software overhead. Additionally, we apply the MBRL approach from `PlaNet' \cite{PlaNet}, attempting to learn a dynamics model for the Duckietown simulation, and compare its performance on the Duckietown and  `Cheetah Run' environments. The code can be found at: \\ \lstinline{https://github.com/alik-git/mbrl-lib}

% \\ In this work, we apply model based reinforcement learning techniques to the Gym-Duckietown environment. Specifically, we use the approach proposed in the `PlaNet' paper \cite{PlaNet} that attempts to learn a dynamics model of the environment in the latent space. 

% \\ \\  This family of algorithms has
% many subcomponents that need to be carefully selected and tuned. As a result the
% entry-bar for researchers to approach the field and to deploy it in real-world tasks
% can be daunting

% \\ \\ To be modified
% Why is it important 
% What did you tried
% What are the results
\end{abstract}

\vspace{-0em}

\section{Introduction }\label{intro}


{\crr{1 [4 pts] Project Introduction (2 paragraphs, hopefully a
figure)

Tell your readers why this project is interesting. What we can learn. Why it is an important
area for robot learning.

} }

See Abstract. This project is interesting because we would like to have robots perform difficult tasks within large environments with complex dynamics. MBRL is still very difficult for such tasks since the learned world model must accurately represent and make predictions about the environment. Model free (MF) approaches can perform better on such tasks, but they typically require an amount of interaction with the environment that is very inefficient when compared to MB approaches. 

Therefore, improving MBRL approaches to outperform MF approaches for tough environments would make it easier to develop robot solutions for the tasks we care about, since the barrier to entry posed by the inefficient sample complexity would be eliminated. Additionally, the best learned world models could be reused for a variety of tasks. 

In this work we attempt to use two MBRL algorithms, Dreamer-v1 and PlaNet, to learn a world model for the Gym-Duckietown environment, which we then use to train a lane-following policy. The Gym-Duckietown environment is significantly larger and more complex than environments included in the aforementioned algorithms' respective papers. We also provide the code we used, where we place a strong emphasis on documentation and modularity, so that more MBRL approaches can be applied to Gym-Duckietown, where our Dreamer-v1 and PlaNet can serve as comparative baselines. This helps us move toward our goal of getting robots to perform the kinds of difficult tasks we care about.

\begin{figure}[t]
\begin{center}
\begin{subfigure}{.25\textwidth}
  \centering
  \includegraphics[trim={0 0 0 0cm}, clip, width=0.90\textwidth]{cheetah_env.jpg}
  \caption{HalfCheetah-v2}
  \label{fig:fig1a}
\end{subfigure}%
\begin{subfigure}{.25\textwidth}
  \centering
  \includegraphics[width=0.97\textwidth]{duckietown_POV.png}
  \caption{Gym-Duckietown}
  \label{fig:fig1b}
\end{subfigure}
\end{center}
\begin{subfigure}{0.5\textwidth}
    \includegraphics[trim={0cm 9cm 0 0.5cm},clip, width=\textwidth]{duckietown_bird_view.png}
    \caption{Overhead view of the Gym-Duckietown environment}
    \label{fig:fig1c}
\end{subfigure}
\caption{Images of the environments we used for this work. For each, we use the PlaNet algorithm to learn a dynamics model and train a reinforcement learning agent.}
\label{fig:fig1}

\end{figure}

\pagebreak

{\crr{
2 [2 pts] What is the related work? Provide references:
(1-2 paragraph(s))

Give a description of the most related work. How is your work similar or different from the
most related work? 
} }

{\crr{
3 [2 pts] What background/math framework have you
used? Provide references: (2-3 paragraph(s))

Describe what methods you are building your work on. Are you using RL, reset-free, hardware, learning from images? You want to provide enough information for the average student
in the class to understand the background. 
} }

The following section is the answer for both question 2 and question 3. The majority of this answer was taken directly from Ali's previous project since the related work has not changed. We have added a section for Dreamer.


% intro, int, chl (challenges), exp (experiments), disc (Discussions), conc (conclusion), liam (Liam's recommendation)

{\cgg {

\subsection{Model Based vs Model Free RL}%(Liam)some motivation about  why  you  thought  it  was  a  good  idea  to  try
Reinforcement learning is an active field of research in autonomous vehicles. Model-based approaches are used less frequently than model-free approaches since model-free approaches have had better performance in the past and as a result it is easier to find existing implementations. 

The largest advantage that model based approaches offer is their superior sample complexity. That is, model based approaches can use orders of magnitude less data or interaction with the external environment compared to model-free methods. This is because model based methods can train the policy on the internal model of the environment as opposed to the external environment itself. Also, model-free methods implicitly learn a `model of the environment' in some way eventually in order to predict the long-term reward of an action, they just do so inefficiently. Additionally, another advantage that model-based methods have is that the learned model of the environment can be task agnostic, meaning that it can be used for any task that requires predicting the state of the environment in the future.
% Model-base biggest advantage is that the model helps to use more efficiently the data and reduce computation time where in model-free, the model is learned implicitly which cause a waste of computation resources.




% \vspace{6em}
% using latent space \citep{PlaNet}. Latent space augments the information gain since the latent space is a dense representation of the input. A great effect of  this method is that it reduces computation time since the dimension of the input is reduced, making the model-base main advantage(computation time) even better. In the PlaNet paper, the computation efficiency gain was in the area of two orders of magnitude and the results were close to model-free and even better on some tasks. 

% Recent improvements in model-based approaches have made the model-based more competitive by using latent space \citep{PlaNet}. Latent space augments the information gain since the latent space is a dense representation of the input. A great effect of  this method is that it reduces computation time since the dimension of the input is reduced, making the model-base main advantage(computation time) even better. In the PlaNet paper, the computation efficiency gain was in the area of two orders of magnitude and the results were close to model-free and even better on some tasks. 

% In RL, either you train in the real world which is often really slow, or you train in a simulator which is much faster, but is dependant on the quality of the simulator and on the transfer back into the real world.It is more computationally heavy too since you have to model the real world in addition to you're RL models.

\subsection{MBRL-Lib}

There are many available open source implementations of popular \mf approaches \cite{baselines, pytorchrl} and \mb approaches \cite{bacon}. During our review, we found the MBRL-Lib toolbox \cite{MBRL} to be most useful. 

MBRL-Lib contains implementations of various popular \mb approaches, but more importantly it is designed to be modular and compatible with a wide array of environments; and makes heavy use of configuration files to minimize the amount of new code needed to tweak an existing approach. Integrating the Gym-Duckietown environment to work with this toolkit would allow users to easily experiment with and evaluate a variety of model based RL approaches in the Duckietown simulation.

\subsection{Gym-Duckietown}
\label{sec:gd}
The environments in which to deploy RL methods can be as simple as tic-tac-toe or as complex as a nearly photo realistic simulation of a whole city \cite{carla}. The most popular environments for current RL research are those provided in the OpenAI Gym toolkit \cite{openaigym}. OpenAI Gym also provides a set of standards which can be used to make environments widely compatible with any software designed to accommodate those standards. MBRL-Lib is an example of such software, and comes with the standard environments from OpenAI Gym built in. 

Naturally, this makes Gym-Duckietown---a self-driving car simulator for the Duckietown universe already built as an OpenAI gym environment \cite{dtown}---the perfect candidate to make available for use with the \mb approaches provided by MBRL.

This is not to say however, that Gym-Duckietown works perfectly with MBRL-Lib right out of the box. Duckietown has significantly more complex dynamics than the standard \og environments. Consider for example that the Cheetah environment (See Fig. \ref{fig:fig1a}) only consists of one (albeit complex) object in a plain background with the camera always tracking it. Compared to \gd, where the camera is fixed on the car which moves through the scene, drastically changing the objects found in different observations.
Our results indicate that while MBRL methods have the potential to perform well in \gd, they need to be carefully tuned and modified to achieve results comparable to those of the current baselines.

% None of the standard OpenAI Gym environments 

% The Duckietown gym is a great simulator, but it's a big one, much bigger than the one needed for the cheetah-run task in PlaNet. It is then much slower too. To run a reinforcement algorithm we need to use the simulator a lot and it will consume a vast amount of computation resources. It is then useful to experiment with model-based algorithms learning from the latent space to save as much computation resources as possible.

\subsection{PlaNet}

The reason \mb approaches have often struggled to match the performance of \mf approaches despite their superior sample complexity is that it is difficult to accurately learn a dynamics of a large and complex environment. The effects of lighting, contact dynamics, non-linear motion etc. are nearly impossible to account for explicitly in enough detail for large scenes. This has prevented \mb approaches from accurately predicting states many steps into the future.

Recent advances in model-based approaches have made them more competitive by leveraging a latent space representation of the world. By using an encoder network to map the environment's high dimensional observations into low dimensional embeddings and learning the dynamics model in the latent space, approaches like PlaNet  \cite{PlaNet} can be used to learn a dynamics model capable of accurate predictions for a complicated environment. This also has significant computation efficiency gains, as it is much faster to propagate dynamics forward in the latent space as opposed to in an explicit simulation. The core of PlaNet's approach also involves both deterministic and stochastic transition components and a multi-step variational inference objective that they call `latent overshooting'.

In the original paper, the authors evaluate PlaNet on environments in the DeepMind control suite \cite{dmc}, but these environments are also available as part of OpenAI Gym, which is what MBRL-Lib and this project use for evaluation.

} } 

\subsection{Dreamer}

To observe how Duckietown scales with different and more performant MBRL algorithms and to facilitate learning, one of the goals in this project is to develop an implementation of Dreamer \citep{Dreamer}, which does not exist in MBRL-Lib. Similar to PlaNet, Dreamer uses a recurrent state space model (RSSM) to represent the underlying MDP with partial observability. Where it differs from PlaNet is in the model fitting section and the planning section. With Dreamer, the model fitting part is broken into a dynamics learning section and a behavior learning section which does rollouts of imagined trajectories, and finally updates parameters for the action model, and value model using gradients of value estimates for the learning objective. Since these trajectories are imagined, the authors utilize reparameterization for continuous actions and states. Lastly, Dreamer computes the state from history during the environment interaction step, with noise added to the action for exploration.

In the paper, the authors run Dreamer on the DeepMind control suite, similar to PlaNet. However, their implementation is in TensorFlow, and will need to be re-implemented in PyTorch for direct comparison.

{\crr {4 [4 pts] Project Method, How will the method work (1-2
pages + figure(s)/math)

Describe your method. Again, You want to provide enough information for the average
student in the class to understand how your method works. Make sure to include figures

} }




% The format of this paper is as follows: 
% % In Sec. \ref{intro} we briefly cover the relevant background / related works for this project, namely PlaNet, \ml and \gd. 
% % In Sec. \ref{int} we cover our code that integrates \gd \ with \ml and its functionality. 
% % In Sec. \ref{chl} we discuss the main hurdles we ran into with this project. 
% % In Sec. \ref{exp} we showcase some experiments using PlaNet approach in \gd \ and the consequent results. 
% % Finally, in Sec. \ref{disc} and  Sec. \ref{con} we reflect on the project as a whole and discuss the potential for future work in this area.
% \begin{itemize}
%     \item In Sec. \ref{intro} we briefly cover the relevant background / related works for this project, namely PlaNet, Dreamer, \ml and \gd. 
% \item In Sec. \ref{int} we cover our code that integrates \gd \ with \ml and its functionality, along with the beginnings of the Dreamer implementation.
% \item In Sec. \ref{chl} we discuss the challenges we encountered. % with this project. 
% \item In Sec. \ref{exp} we showcase some experiments using the PlaNet approach and the Dreamer implementation in \gd \ and our results. 
% \item Finally, in Sec. \ref{disc} and  Sec. \ref{con} we reflect on the project as a whole and discuss the potential for future work in this area.
% \end{itemize}







\section{Applying PlaNet and Dreamer to Gym-Duckietown}\label{int}


% \subsection{Changes to \ml}

% As noted before, the \ml code is well organized, and adding the \gd \ environment as a configurable option was straightforward. We've included the code here:

% \begin{lstlisting}[language=Python, caption=Adding Gym-Duckietown to MBRL-Lib, label=lis1]
% # Use the duckietown environment with 
% # settings that closely match the settings 
% # used for cheetah run
% if cfg.overrides.env=="duckietown_gym_env":
%     env = mbrl.env.DuckietownEnv(
%         domain_rand=False, 
%         camera_width=64, 
%         camera_height=64
%     )
%     term_fn = \
%     mbrl.env.termination_fns.no_termination
%     reward_fn = None
% \end{lstlisting}

% The main pre-requisite to being able to do this is to import the Gym-Duckietown code as an external python module into \ml so that the \lstinline{DuckietownEnv} object can be used directly without creating any new classes or objects. We took efforts throughout this project to avoid re-writing any code already found in either code base, and to import and use the original code directly. 

% Next we add a configuration file for default parameter values associated with the Duckietown environment. This is a YAML file called \lstinline{planet_duckietown.yaml} and for now only contains default values for the PlaNet algorithm, however more config files can be easily created for other algorithms as long as the follow the same format. 

% We ran into some issues with using the observations from \lstinline{DuckietownEnv} with the PlaNet algorithm, we expand on this in the Challenges section.

% % We developed a  bridge between the Facebook-research \cite{MBRL} and the PlaNet paper \cite{PlaNet} to allow for latent space planning from pixel using the Duckietown gym. 

% % The basic idea was to find a way to allow future research in latent space for reinforcement algorithm. We presented the PlaNet paper and it was our baseline algorithm. We found a well written library called MBRL\citep{MBRL} that is a tool for Model-Based Reinforcement Learning. Since it was easy to read(for a code of that size), we decide to use their code instead of the PlaNet paper one and conveniently, the MBRL paper already had a support for PlaNet's algorithm. MBRL was using the openAI gym environement, so we had to accommodate MBRL with Gym Duckietown to them work together.

% \subsection{Changes to \gd}

% Other than a couple of minor bug fixes and tweaks we did not need to make any changes to \gd \ to use it with \ml. 
% We turn off domain randomization and change the default observation size (in \ls{[height, width]}) to be  \ls{[64,64]} as opposed to \ls{[640, 480]} as we wanted to closely match the hyper-parameters used for the PlaNet on the HalfCheetah-v2 environment as a starting point. Another practical issue is simply that using observations of size \ls{[640,480]} may require a very large amount of video memory (up to 3 TB) depending on the chosen batch size. To be clear, the choice of size \ls{[64,64]} is arbitrary and is subject to future experimentation.

% \subsection{Integrating W\&B}

% \ml uses Hydra \cite{hydra} for logging and organizing runs. They also have some basic visualization tools specific to MBRL techniques that they have built into their library. These tools however still lack the ability to automatically plot metrics for each run, and back up metrics off-site, filtering runs by a certain metric, and it can be quite cumbersome to wade through hundreds of folders in the Hydra logging directory to find and interpret your results.

% To alleviate this issue, we import and use the Weights \& Biases library \cite{wandb} to track our runs. We also log the configuration file, and log a copy of every metric logged by Hydra in MBRL-Lib. This gives an online dashboard that automatically organizes our runs and plots their metrics, as well as all the functions we mentioned earlier. 

\subsection{Model-Based Reinforcement Learning}

To setup the reinforcement learning problem from a model-based reinforcement learning (MBRL) perspective, we adhere to the Markov decision process formulation \citep{bellman1957markovian}, where we use state $s \in \mathcal{S}$ and actions $a \in \mathcal{A}$ with reward function  $r(s,a)$ and the dynamics or transition function $f_\theta$, such that $s_{t+1} = f_{\theta}(s_t, a_t)$ for deterministic transitions, and stochastic transitions are given by the conditional $f_\theta(s_{t+1}\mid s_t, a_t) = \mathbb{P}(s_{t+1}\mid s_t, a_t, ; \theta)$ and learning the forward dynamics is akin to doing a fitting of approximation $\hat{f}$ to the real dynamics $f$ given real data from the system.\\

\subsection{Tuning PlaNet for \gd}

Of the important contributions of PlaNet \citep{PlaNet}, one of them is the recurrent state space model (RSSM). The RSSM has both stochastic and deterministic components and it was shown in PlaNet to greatly improve results compared to purely stochastic or deterministic models on complicated task.
To bring the input images to the latent space, we need an encoder. Since we are using images, a convolution neural net is perfect for the task.\\
No-policy is actually trained since the planning algorithm use only the models to choose the next best action.\\

Since the models are using stochastic decisions, the training is using a variational bound to optimise its parameters. It alternatively optimises the encoder model and the dynamics model by gradient ascent over the following variational bound:
\begin{align*}
    &\ln{p}(o_{1:T}  \mid a_{1:T}) \delequal \ln \int \prod_t p(s_t\mid s_{t-1},a_{t-1})p(o_t\mid s_t)ds_{1:T} \\
    &\geq \sum_{t=1}^{T}  \left(\mathbb{E}_{q(s_t\mid o_{\leq t},a_{\leq t})}[\ln{p(o_t\mid s_t)}]) \right.\\
     - &\left. \mathbb{E}_{q(s_{t-1}\mid o_{\leq t-1},a_{\leq t-1})}[KL[q(s_{t}\mid o_{\leq t},a_{\leq t})\mid \mid  p(s_t\mid s_{t-1},a_{t-1})]] \right)
\end{align*}

The PlaNet models follow a Partially Observable Markov Decision Process(POMDP). It is built on a finite sets of: states($s_t$), actions($a_t$) and observations($o_t$).
\begin{itemize}
    \item Dynamics model : $s_t \sim p(s_t\mid s_{t-1},a_{t-1})$
    \item Encoder : $s_t \sim q(s_t\mid o_{\leq t},a_{\leq t})$
    \item Decoder : $o_t \sim p(o_t\mid s_{t},a_{t-1})$ %like in class not like paper, but makes more sens
    \item Reward : $r_t \sim p(r_t\mid s_t)$
    \item $\gamma$ the discount factor $\gamma \in [0,1]$
\end{itemize}

The rest of the details are outlined for the RSSM representation in comparsion to deterministic and stochastic models is shown in figure \ref{fig:planet_rssm}.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{planet_rssm.PNG}
    % \caption{A comparison of the planet algorithm training reward on the cheetah environment vs on the duckietown gym.}
    \caption{Recurrent state space model comparison to deterministic and stochastic models}
    \label{fig:planet_rssm}
\end{figure}

Although the PlaNet algorithm is already supported and included in \ml, however, it is not simply plug-and-play for use with \gd. Firstly, there are key differences to using \ml algorithms with DeepMind control suites versus the \gd \space environment. \gd \space observations, resets, and the reward function (and tuning), as well as camera orientation in the first person are additional challenges in the \gd \space simulation environment. There are also hyperparameters that influence agent behavior, as well as the planning horizon that have a significant impact on performance. Next, using PlaNet as a world model to decouple planning and acting, PlaNet is used for the Dreamer implementation.

\subsection{Implementation of Dreamer in \ml}

Given that the \ml does not include a Dreamer implementation, but rather the PlaNet model, it makes sense to reuse the recurrent state model from the PlaNet implementation as a world model for Dreamer. However, there are structural components that are missing. For example, in departure from PlaNet, rather than a CEM for the best action sequence under the model for planning, Dreamer uses a dense action network parameterized by $\phi$ and the dense value network parameterized by $\psi$. For the action model with imagined actions, the authors use a tanh-transformed Gaussian \citep{SAC} output for the action network, which provides a deterministically-dependent mean and variance of the state through a reparameterization \citep{kingma2013auto} \citep{rezende2014stochastic} of the stochastic node, adding noise $\epsilon$ back in afterwards (to be inferred).

\begin{equation*} \label{eq:1}
    a_\tau = \tanh(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau) \epsilon), \qquad \epsilon \sim \mathcal{N}(0, \mathbb{I})
\end{equation*}

This formalizes the deterministic output of the action network returns a mean $\mu_\phi(s_\tau)$ and we learn the variance of the noise $\sigma_\phi(s_\tau)$ with this reparameterization, inferring from our normal noise $\epsilon$, to represent our stochastic model.

Then the value network consists of imagined value estimates $V_R(s_\tau) = \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} r_n))$ which is the sum of rewards until the end of a horizon, then using values $v_\psi(s_\tau)$ then computes $V^{k}_N(s_\tau) = \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{h = \min(\tau+k, t+H)-1} \gamma^{n-\tau}r_n + \gamma^{h-\tau} v_\psi(s_h)))$ as a estimate of rewards beyond $k$ steps with the learned value model, and $V_\lambda(s_\tau)$ which is a exponentially weighted average of $V^{k}_N(s_\tau)$ at different values of k, shown in the following. 

$$V_\lambda(s_\tau) = (1 - \lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V_{N}^{n}(s_\tau) + \lambda^{H-1}V_{N}^{H}(s_\tau)$$

This helps Dreamer do better with longer-term predictions of the world, over shortsightedness with other types of dynamics models for behavior learning. Since Dreamer disconnects the planning and action by training an actor and value network and uses analytic gradients and reparameterization, it is more efficient than PlaNet which searches the best actions among many predictions for different action sequences. This motivates the implementation of Dreamer to compare to PlaNet with potential performance improvements with a similar number of environment steps. The policy is trained via using the analytical gradient $\nabla_\phi \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} V_\lambda(s_\tau))$ from stochastic backpropagation, which in this case becomes a deterministic node where the action is returned shown in \ref{eq:1}, with the value network being updated with the gradient  $\nabla_\psi \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} \frac{1}{2}\mid \mid v_\psi(s_\tau) - V_\lambda(s_\tau))\mid \mid ^2$ after imagined value estimates are computed. All of this happens in the update steps for behavior and dynamics learning. Finally, in an environment interaction time step, the agent gets states from its history and returns actions from the action network, and value model estimates the imagined rewards that the action model gets in each state. These are trained cooperatively in a policy iteration fashion.

{ \cgg { 
\subsection{Documentation}

We want to make using our code as straightforward as possible. As such, we have provided a detailed read-me file at: \mbox{\lstinline{https://github.com/alik-git/mbrl-lib}}

It includes installation instructions---including those for dependencies---along with usage instructions and example commands. We have also included troubleshooting advice for common problems we encountered as well as a debugging configuration setup for those using the VSCode editor.

} } 

% In order to easily visualize 

% %(Liam)whatworked and how you know it worked (results)
% We managed to make the translation between the Duckietown gym and the MBRL framework work. We arranged the code to be easily reusable and documented every step to allow for future reinforcement learning research in Duckietown to be quickly set up.\\
% With training and fine tuning, things can get messy really fast. We incorporated the "wanb" library to manage efficiently the data and visualisation.\url{https://docs.wandb.ai/quickstart}
% It is a nice tool that automatically saves data and lets you manage it and visualise it easily.


\section{Challenges}\label{chl}

{\crr{
5 [2 pts] What new skills are you(s) learning from this
project?

List some of the technical skills you are learning from studying this method.
}}

\subsection{New Skills Learned}\label{ns}

To set up good experiments for this project, we had to change the reset function and the reward function of \gd. Spending time working on these challenges forced us learn new skills for designing good reward functions, and setting up good experiments that are both fair and allow the policy to learn good behavior. We also had to log a significant amount of information for our project, including videos of the policy, which posed an organizational challenge, for example making sure that each video is saved with a corresponding configuration file detailing how much training the policy had, how much reward was achieved in the video, etc. 

Implementing a our own version of the Dreamer algorithm within the MBRL-Lib repo forced us to understand and think about what role each component plays, how to make the parameters of the algorithm configurable, and overall how to implement algorithms in a way that is flexible for future changes and improvements.

Overall, this project allowed us to learn a lot about why MBRL struggles with large environments, the common problems associated with experiments design, reward functions, and doing RL research in general. Each member of team feels a lot more confident in approaching an RL project on their own. 

\subsection{Technical Issues}\label{ti}

The reward function was originally specified with negative rewards for driving on the wrong side of the road, which was found to be a compounding issue with random spawning. Frequently, the agent would spawn in locations with no way to obtain positive rewards, and the episode terminates when the agent goes off track. So, the agent would avoid the negative rewards from a bad spawning location and often go straight off-track. To stabilize learning, the agent has fixed spawn locations and the reward is now tuned to be function of the distance from the center of the proper lane. It was also discovered that there are multiple wrappers doing pre-processing on the observations for stacking, which means exporting these as "raw images" is not possible without unrolling all of the wrappers given the continuous control suites, making it difficult to save videos of policy behavior and evaluate learned policies outside of \ml.

% \subsection{Observation shape mismatch}

% The most challenging issue to make \gd \ work with \ml was that the PlaNet encoder model expects the observations to be of the following shape: \ls{[batch_size, color_channels, image_width, image_height]}. However the \ls{DuckietownEnv} returns observations in the shape \ls{[image_width, image_height, color_channels]}. The problem here is that the PlaNet encoder expects \ls{color_channels} to be before \ls{image_width} and \ls{image_height}. We found no way to adjust the the model to accomodate this shape via the config files, nor a way for \gd \ to return observations with the dimensions permuted. As a result, we were forced to manually permute the observation Tensor before it is fed to the PlaNet encoder , and permute it back again after the resulting embedding is fed through the PlaNet decoder. These modifications were made in the \ls{planet.py} file in the \ls{mbrl/models} folder.

% A better way to do this would be to have pre-processing function for the \gd \ observations, but since the \ml code is designed to be quite modular, we could not find a clean way to add such a function without disturbing some other part of the code.
{\cgg {
\subsection{Importing \gd }
Since \gd \ itself is not a pip package one can import from PyPi, we have included instructions for how to manually import it in the \lstinline{README.md} file for our project. 
} }

% \subsection{Hyperparameter Tuning}
% \label{hp}

% As noted earlier, MBRL methods are quite sensitive to the choices of hyperparameters and the environments we are using are complex, so training times can be significant. None of the configurations we tried achieved consistently positive rewards, even when left to train overnight. Therefore, for the scope of this project we chose to use the same hyperparamters that the PlaNet authors had success with the HalfCheetah-v2 environment, which we consider to be our baseline for comparison.

% In our results (See Fig. \ref{fig:reward}, \ref{fig:loss}), we observe a positive trend in both the loss curve and the reward curve overtime for the \gd \ environment, although as expected the overall results are much better for the HalfCheetah-v2 environment, which the parameters are designed to suit. These results can be treated as a starting point for model based reinforcement learning on the \gd \ environment.

% \subsection{Other Misc Minor Issues}

% There were a handful of other minor issues we had to solve for the project. These included:
% \begin{itemize}
%     \item Commenting out / updating some deprecated code in \gd.
%     \item Fixing an off by one error in the sequence transition sampler in \ml that would try to sample a transition even when the current batch was empty.
%     \item Creating a copy of the configuration object that was not nested, because that would prevent W\&B from being able to parse it.
%     % \vspace{em}
%     % \item Fixing the model and saving functionality in \ml that would cause crashes if the destination folder didn't already exist.
% \end{itemize}

\pagebreak
Continued on next page
\pagebreak

\section{Experiments}\label{exp}



{\crr{
6 [6 pts] Experiments

In this section

} }

{\crr{

6.1. Describe what experiment(s) you are going to run and why? How do these show you
have met your learning goals?
} }

We will include the training curves for both Planet and Dreamer with various hyperparameters on Gym-Duckietown. This will give the reader a feel for how much reward the policy can get.

{\crr{
6.2. What do you think the results of these experiments will be?
} }

We think that there will be an eventual upward trend of the "average episode" reward. The absolute quantity of reward may be low, since the environment is tough to learn, and the reward function for \gd is not perfect, but our goal is demonstrate that at least some good driving behavior is being learned. In addition, it is likely that increasing batch sizes, sequence lengths for the horizon, and also reduced learning rates with a more stable reward function will net benefits on the PlaNet agent. With Dreamer, it will likely be sensitive to some of the same parameters because it uses PlaNet as a world model, but will also be sensitive to the action noise and also number of k steps for imagined value estimate lengths. The dense networks from Dreamer might also benefit from additional layers, given that the original paper used on DeepMind control suites uses 3 layers of 300 ELU activations, the representation model uses the same architecture as the CNN encoder/decoder from the World Model paper \citep{ha2018world}. 

\begin{figure}[h]
    \centering
    \includegraphics[trim={0cm 0cm 0 0.5cm},clip, width=0.5\textwidth]{param_search.png}
    % \caption{A comparison of the planet algorithm training reward on the cheetah environment vs on the duckietown gym.}
    \caption{Preliminary Experiment: A hyperparameter search of different learning rates on short runs of PlaNet on \gd. Some large learning rates cause the weight matrices to overflow and result in a the "null" reward.}
    \label{fig:param_search}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[trim={0cm 0cm 0 0.5cm},clip, width=0.5\textwidth]{param_search_zoomed_in.png}
    % \caption{A comparison of the planet algorithm training reward on the cheetah environment vs on the duckietown gym.}
    \caption{Preliminary Experiment: A hyperparameter search of different learning rates on short runs of PlaNet on \gd. Zoomed in to show more detail for learning rates between 0.12 - 0.13}
    \label{fig:param_search_z}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[trim={0cm 0cm 0 0.5cm},clip, width=0.5\textwidth]{planet_onduck_reward.png}
    % \caption{A comparison of the planet algorithm training reward on the cheetah environment vs on the duckietown gym.}
    \caption{Preliminary Experiment: This serves as an example of what the learning curve could look like for PlaNet on the \gd  environment. At the moment, various issues prevent the policy from performing well, but we are working on fixes. (better reward function, fixed spawns, better hyperparameters) }
    \label{fig:param_search_z}
\end{figure}

{\crr{
6.3. Sketch out the figures that you will later generate from your work. Spend a few
minutes drawing them out in GIMP or photoshop. Why will these be enough evidence
for learning? Is anything missing?
} }





{\crr{
Keep in mind these experiments are for this course project. What is expected is that you
should provide evidence that your method works and it has been coded up well. Provide
evidence of this via your data and learning graphs. However, this should not be restricted
to learning graphs.
} }

\pagebreak



{\crr{
7 [2 pts] Video Results

Include a link to a video of the method so far. This is not intended to be a final performance
video. Provide something to help me give feedback on what I would do to improve this
method or debug issues.

} }

Preliminary videos of PlaNet running on \gd can be found here: \href{https://imgur.com/a/QpOdMjL}{https://imgur.com/a/QpOdMjL}

% As noted in Sec. \ref{hp}, we used the same PlaNet hyperparameters for \gd \ that the PlaNet authors chose for the `Cheetah Run' task on the HalfCheetah-v2 environment. Our results are shown in Fig. \ref{fig:reward} and Fig. \ref{fig:loss} where we observe that while the PlaNet model performs significantly better on the HalfCheetah-v2 environment, it still improves overtime on the \gd \ environment. 

% Each of the runs shown took approximately 10 hours on an NVIDIA RTX 2070 GPU. In the \hc \ environment the PlaNet model can achieve positive rewards from the beginning whereas for the \gd \ environment it takes a significant amount of time for the average reward to be positive. We did not have time within the scope of this project to have runs long enough to consistently observe positive rewards on the \gd \ environment. 

% However the trend we do observe indicates that given enough time and compute resources, the PlaNet model could learn a useful dynamics model and policy for the \gd \ environment. This may also be explained by the fact that the \gd \ environment is much larger, and has a larger collections of objects than the \hc \ environment. Also the two environments are fundamentally different in that the camera observes the vehicle's point of view and moves freely around the science in \gd \ as opposed to \hc \ where the camera always tracks the cheetah.



% \begin{figure}[t]
%     \centering
%     \includegraphics[trim={0cm 0cm 0 0.5cm},clip, width=0.5\textwidth]{Planet Total Episode Reward Training.png}
%     % \caption{A comparison of the planet algorithm training reward on the cheetah environment vs on the duckietown gym.}
%     \caption{A comparison of PlaNet's Total Episode Reward on the \gd  \ and HalfCheetah-v2 environment on the `Cheetah Run' task.}
%     \label{fig:reward}
% \end{figure}

% \begin{figure}[t]
%     \centering
%     \includegraphics[trim={0cm 0cm 0 0.5cm},clip, width=0.5\textwidth]{Planet Training Observation Loss.png}
%     \caption{A comparison of PlaNet's Training Observation Loss on the \gd  \ and HalfCheetah-v2 environment on the `Cheetah Run' task.}
%     \label{fig:loss}
% \end{figure}
% \begin{figure}[t]
%     \centering
%     \includegraphics[trim={0cm 0cm 0 0.5cm},clip, width=0.5\textwidth]{Planet Total Episode Reward Training.png}
%     % \caption{A comparison of the planet algorithm training reward on the cheetah environment vs on the duckietown gym.}
%     \caption{A comparison of PlaNet's Total Episode Reward on the \gd  \ and HalfCheetah-v2 environment on the `Cheetah Run' task.}
%     \label{fig:reward}
% \end{figure}

% On figure 2 we can observe some proof of successful training where the reward gets better and better with time. It seams like the reward is much better with the cheetah experiment, but it might be caused by the different reward function in both environment. If Duckietown's reward is more severe than cheetah's one it might get lower reward for the same quality of model. Furthermore, autonomous driving being a harder task than the cheetah run, it might only need more training.


\section{Discussion}\label{disc}

{\crr{

9 [2 pts] How is the timeline for the project progressing?
At least month-by-month granularity, ending with a
complete project (outline what has been completed so
far).


} }

We are definitely a little behind, but are happy we the progress and results we have so far. We have learned a lot about MBRL.

\begin{enumerate}
    \item \textbf{Feburary} \\
    Write boilerplate and infrastructure code for performing experiments.
    \begin{enumerate}
        \item \textbf{(DONE)} Implement boilerplate code to visualize learned policies in Duckietown. (i.e take a saved model parameter file, load it into a model, have it pick actions and pass them into duckietown and render the observations) \hl{This will help us visualize the behavior of our policy to ensure it is working the way we expect, and also help us diagnose and fix bugs.}
        \item \textbf{(DONE)} Implement model caching during training for better logging.
        \item BONUS: Try out one other algorithm than PlaNet and get a feel of its performance.
    \end{enumerate}
    \item \textbf{March} \\
    Start hyper parameter tuning to see what performance is possible
    \begin{enumerate}
        \item Train all models at least once to get raw baseline performance, then pick one model for hyper-parameter tuning for the Duckietown environment. 
        \item \textbf{(DONE - for PlaNet)} \hl{Debugging the model: For both Dreamer and PlaNet, we plan to log and visualise quantities like observations, the distribution of the actions outputted by the model etc. This will (hopefully) allow us to gain insights into how the model is working (correctly or incorrectly) and then diagnose and fix issues.}
        \item Do hyperparameter tuning on said model.
        \item Try domain randomization to see if good performance can still learned with domain randomization.
    \end{enumerate}
    \item \textbf{April} \\
    Evaluate obtained results, write a report, make figures, and add documentation.
    \begin{enumerate}
        \item Freeze experiments and record performance 
        \item Write the report analyzing our method and results.
        \item Write about promising directions for future experiments and areas of inquiry. 
        \item Make project page with documentation of how to use our work
        \item BONUS: Run real robot experiments on Duckiebots or another robot.
    \end{enumerate}
    

\end{enumerate}



{\crr{

10 [2 pts] How is the work divided?

Provide a description on what each group member is working on as part of the project. I
recommend each student work on most of the parts of the project so everyone learns about
the content.

} }

Both members worked on everything.

Student Name: Ali focused more on infrastructure code for \gd and logging.

Student Name: Paul focused more on the Dreamer implementation.




% Our hope for this project is that it may be used as a starting point for more experimentation with MBRL methods in the \gd \ environment. Our results indicate that there is certainly potential for significant performance gains by (1) more compute resources, and (2) hyperparameter tuning for PlaNet to better suit the \gd \ environment. There is also the option of trying different approaches all-together other than PlaNet such as PETS \cite{pets} and MPBO \cite{mbpo}, these are already implemented in \ml and therefore are great candidates for experimentation.



\section{Conclusion}\label{con}

{\crr{

8 [2 pts] Conclusions

What have you learned? What would you do differently next time? Reflect on the scope of
your project, was it too much? Why?

}}

In this work we applied two MBRL approaches to the Duckietown environment. We learned that MBRL models are tough to tune, and there are so many details that matter like the reset and reward function of the environment that make RL quite difficult. What we would do differently next time: Ask for help earlier, lots of people know much more about Duckietown than we do and were very helpful, I imagine it is the same for MBRL-Lib, so I would have reached out to the authors earlier. Trying to fit in real robot experiments was definitely too much. Both \gd and MBRL pose significant challenges to work with as it is. The experiments are quite time consuming. They require are time span of months and not weeks.

% In this work, we integrate the \gd \ environment into the \ml toolbox to enable easy experimentation of MBRL approaches for the autonomous driving task within the Duckietown simulator. Our code is open source and well documented, and we provide thorough instructions on how to setup and run it. We also showcase some preliminary experiments applying the PlaNet approach to the \gd \ environment with results that indicate the potential for future MBRL research to achieve significant performance improvements on the autonomous driving task.

% % Figure 3 highlights the fact that the model does not optimizes as well for the duckietown environment, but for the same reasons as for the reward it might simply be due to the difference between environment. We hope that the convergence around a loss of 50 can be brought down by fine tuning the hyperparameters.\\

% % \subsection{sub Challenges}
% % Training a model that actually works well in simulation has proven difficult. There are multiple possible reasons for that. Autonomous driving is a pretty complex problem thus it is hard to come with a reward function that really helps learning the dynamics. Even with a great reward function, the amount of training data necessary for a good exploration of the action space is so big that there was no chance for it to run on our computer and even for researcher with more resources it might becomes a problem at some point. \\

% % Fine tuning is an other problem when to train a model takes all night and that only outputs results for 1 hyper parameter change. With a good protocol to test and a lot more time than we had, it would have been possible.Still, we tried some values for gradient clipping and saw some good improvements(***do we have the numbers?). This indicates that it might just be a fine tuning problem to get a good model for the task.

    
% % \end{item}
    
% % \end{enumerate}

 

% % We created a recipe for anyone in the futur that would want to set this up in Duckietown. Our main problems were:

% % images from duckie town are [620,480] and we quicly ran out of memory on our computer, we decided to shrink them into [64,64] images, but this is probably too low quality for the model to learn the dynamics. With better resources it would be interesting to augment the images resolutions to get better results.
% % % \subsection 
% % %idea is there 
% % MBRL was expecting the first dimension to be the number on input(3 for rgb), so the inputs had to be [3,64,64], but Duckietown gym after lowering the resolution was in the [64,64,3] format. There was no easy way to fix this as each time we tried modifying the order somewhere in the code, it would break things everywhere else. We found that by translating before the encoder and doing the same process backward after the decoder works because we haven't change anything according to the rest of the code.





% % \section{Their code }
% % \section{Our code/Methodology}

% %(Liam)anytake away messages and conclusions or ideas to overcome the problems you encountered.

% % ***Weird colors from Duckietown\\
% % %We did XYZ, which resulted in XYZ, this allows future work XYZ to be done.
% % We put together a baseline for future reinforcement learning work using the duckieTown simulator. It acheived some basic training and might be great with better computation resources or just with more fine tuning. ***\citep{TIA}

% % \section{Liam's recommandation}
% % \label{liam}
% % Some background about your subject (not nearly as detailed as the presentation - but more specifically related to what you tried), a description of what you tried, some motivation about why you thought it was a good idea to try, what worked and how you know it worked (results), what didn't work, any hypotheses about why things didn't work, any take away messages and conclusions or ideas to overcome the problems you encountered.

% % \subsection{PlaNet Methodology}%(Liam)Some background about your subject

% % Recurrent state space models have both stochastic and deterministic components and it was shown in PlaNet to greatly improve results compared to purely stochastic or deterministic models on complicated task.
% % To bring the input images to the latent space, we need an encoder. Since we are using images, a convolution neural net is perfect for the task.\\
% % No-policy is actually trained since the planning algorithm use only the models to choose the next best action.(see \citep{PlaNet})\\

% % Since the models are using stochastic decisions, the training is using a variational bound to optimise its parameters. It alternatively optimises the encoder model and the dynamics model by gradient ascent over the following variational bound:
% % \begin{align*}
% %     &\ln{p}(o_{1:T}  \mid a_{1:T}) \delequal \ln \int \prod_t p(s_t\mid s_{t-1},a_{t-1})p(o_t\mid s_t)ds_{1:T} \\
% %     &\geq \sum_{t=1}^{T}  \left(\mathbb{E}_{q(s_t\mid o_{\leq t},a_{\leq t})}[\ln{p(o_t\mid s_t)}]) \right.\\
% %      - &\left. \mathbb{E}_{q(s_{t-1}\mid o_{\leq t-1},a_{\leq t-1})}[KL[q(s_{t}\mid o_{\leq t},a_{\leq t})\mid \mid  p(s_t\mid s_{t-1},a_{t-1})]] \right)
% % \end{align*}
% % \subsubsection{POMDP}
% % The PlaNet models follow a Partially Observable Markov Decision Process(POMDP).It is build on a finite sets of: states($s_t$), actions($a_t$) and observations($o_t$).
% % \begin{itemize}
% %     \item Dynamics model : $s_t \sim p(s_t\mid s_{t-1},a_{t-1})$
% %     \item Encoder : $s_t \sim q(s_t\mid o_{\leq t},a_{\leq t})$
% %     \item Decoder : $o_t \sim p(o_t\mid s_{t},a_{t-1})$ %like in class not like paper, but makes more sens
% %     \item Reward : $r_t \sim p(r_t\mid s_t)$
% %     \item $\gamma$ the discount factor $\gamma \in [0,1]$
% % \end{itemize}

\bibliography{references}
\bibliographystyle{icml2018}

\end{document}


% This document was modified from the file originally made available by
% Pat Langley and Andrea Danyluk for ICML-2K. This version was created
% by Iain Murray in 2018. It was modified from a version from Dan Roy in
% 2017, which was based on a version from Lise Getoor and Tobias
% Scheffer, which was slightly modified from the 2010 version by
% Thorsten Joachims & Johannes Fuernkranz, slightly modified from the
% 2009 version by Kiri Wagstaff and Sam Roweis's 2008 version, which is
% slightly modified from Prasad Tadepalli's 2007 version which is a
% lightly changed version of the previous year's version by Andrew
% Moore, which was in turn edited from those of Kristian Kersting and
% Codrina Lauth. Alex Smola contributed to the algorithmic style files.


```


















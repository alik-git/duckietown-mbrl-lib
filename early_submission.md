---
title: Duckietown MBRL-Lib
---


___

**Authors:** [Ali Kuwajerwala](https://alik-git.github.io/), [Paul Crouther](https://www.linkedin.com/in/paul-crouther-47221b52/) <br /> 
**Affiliation:** [Université de Montréal](https://diro.umontreal.ca/accueil/), [Mila](https://mila.quebec/en/) <br />
**Date Published:** April 28, 2022

___




**Abstract**

Model-based reinforcement learning (MBRL) algorithms have various sub-components that each need to be carefully selected and tuned, which makes it difficult to quickly apply existing models/approaches to new tasks. In this work, we aim to integrate the Dreamer algoithm into an existing popular MBRL toolbox, and tune the Dreamer-v1 and PlaNet algorithms to the Gym-Duckietown environment. We also provide trained models and code to to use as a baseline for further development. Additionally, we propose an improved reward function for RL training in Gym-Duckietown, with code to allow easy analysis and evaluation of RL models and reward functions.

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



The environments in which to deploy RL methods can be as simple as tic-tac-toe or as complex as a nearly photo realistic simulation of a whole city \cite{carla}. The most popular environments for current RL research are those provided in the OpenAI Gym toolkit \cite{openaigym}. OpenAI Gym also provides a set of standards which can be used to make environments widely compatible with any software designed to accommodate those standards. MBRL-Lib is an example of such software, and comes with the standard environments from OpenAI Gym built in. 

Naturally, this makes Gym-Duckietown---a self-driving car simulator for the Duckietown universe already built as an OpenAI gym environment \cite{dtown}---the perfect candidate to make available for use with the \mb approaches provided by MBRL.

This is not to say however, that Gym-Duckietown works perfectly with MBRL-Lib right out of the box. Duckietown has significantly more complex dynamics than the standard \og environments. Consider for example that the Cheetah environment (See Fig. \ref{fig:fig1a}) only consists of one (albeit complex) object in a plain background with the camera always tracking it. Compared to \gd, where the camera is fixed on the car which moves through the scene, drastically changing the objects found in different observations.
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

To setup the reinforcement learning problem from a model-based reinforcement learning (MBRL) perspective, we adhere to the Markov decision process formulation \citep{bellman1957markovian}, where we use state $s \in \mathcal{S}$ and actions $a \in \mathcal{A}$ with reward function  $r(s,a)$ and the dynamics or transition function $f_\theta$, such that $s_{t+1} = f_{\theta}(s_t, a_t)$ for deterministic transitions, and stochastic transitions are given by the conditional $f_\theta(s_{t+1}|s_t, a_t) = \mathbb{P}(s_{t+1}|s_t, a_t, ; \theta)$ and learning the forward dynamics is akin to doing a fitting of approximation $\hat{f}$ to the real dynamics $f$ given real data from the system.\\

### PlaNet for Gym-Duckietown

Of the important contributions of PlaNet \citep{PlaNet}, one of them is the recurrent state space model (RSSM). The RSSM has both stochastic and deterministic components and it was shown in PlaNet to greatly improve results compared to purely stochastic or deterministic models on complicated task.
To bring the input images to the latent space, we need an encoder. Since we are using images, a convolution neural net is perfect for the task.\\
No-policy is actually trained since the planning algorithm use only the models to choose the next best action.\\

Since the models are using stochastic decisions, the training is using a variational bound to optimise its parameters. It alternatively optimises the encoder model and the dynamics model by gradient ascent over the following variational bound:
<!-- $$ \ln{p}(o_{1:T}  |a_{1:T}) \delequal \ln \int \prod_t p(s_t|s_{t-1},a_{t-1})p(o_t|s_t)ds_{1:T} \\
&\geq \sum_{t=1}^{T}  \left(\mathbb{E}_{q(s_t|o_{\leq t},a_{\leq t})}[\ln{p(o_t|s_t)}]) \right.\\
- &\left. \mathbb{E}_{q(s_{t-1}|o_{\leq t-1},a_{\leq t-1})}[KL[q(s_{t}|o_{\leq t},a_{\leq t})|| p(s_t|s_{t-1},a_{t-1})]] \right $$  -->

The PlaNet models follow a Partially Observable Markov Decision Process(POMDP). It is built on a finite sets of: states($s_t$), actions($a_t$) and observations($o_t$).

###  Model Based vs Model Free RL

The following section is the answer for both question 2 and question 3. The majority of this answer was taken directly from Ali's previous project since the related work has not changed. We have added a section for Dreamer.


Reinforcement learning is an active field of research in autonomous vehicles. Model-based approaches are used less frequently than model-free approaches since model-free approaches have had better performance in the past and as a result it is easier to find existing implementations. 

The largest advantage that model based approaches offer is their superior sample complexity. That is, model based approaches can use orders of magnitude less data or interaction with the external environment compared to model-free methods. This is because model based methods can train the policy on the internal model of the environment as opposed to the external environment itself. Also, model-free methods implicitly learn a `model of the environment' in some way eventually in order to predict the long-term reward of an action, they just do so inefficiently. Additionally, another advantage that model-based methods have is that the learned model of the environment can be task agnostic, meaning that it can be used for any task that requires predicting the state of the environment in the future.

##  Project Method

    4 [6 pts] Project Method, How will the method work (1-2 pages + figure(s) + math + algorithm)
    Describe your method. 
    Again, You want to provide enough information for the average student in the class to understand how your method works. 
    Make sure to include figures, math, or algorithms to help people understand.





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






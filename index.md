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

states($s_t$), actions($$a_t$$) and observations($o_t$)

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

Given that the MBRL-lib does not include a Dreamer implementation, but rather the PlaNet model, it makes sense to reuse the recurrent state model from the PlaNet implementation as a world model for Dreamer. However, there are structural components that are missing. For example, in departure from PlaNet, rather than a CEM for the best action sequence under the model for planning, Dreamer uses a dense action network parameterized by phi and the dense value network parameterized by psi. 

$$a_\tau = ActorNetwork_\phi(s_\tau) \sim q_\phi(a_\tau \mid s_\tau)  \\ v_\phi(s_\tau)  = DenseValueNetwork_\psi(s_\tau) \approx \mathbb{E}_{q(\cdot \mid s_\tau)} \left( \sum_{\tau = t}^{t+H} \gamma^{\tau - t}r_\tau \right)$$

For the action model with imagined actions, the authors use a tanh-transformed Gaussian \citep{SAC} output for the action network, which provides a deterministically-dependent mean and variance of the state through a reparameterization \citep{kingma2013auto} \citep{rezende2014stochastic} of the stochastic node, adding noise $\epsilon$ back in afterwards (to be inferred).

$$a_\tau = \tanh(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau) \epsilon), \qquad \epsilon \sim \mathcal{N}(0, \mathbb{I})$$

This formalizes the deterministic output of the action network returns a mean mu and we learn the variance of the noise sigma with this reparameterization, inferring from our normal noise epsilon, to represent our stochastic model.

$$\text{mean } = \mu_\phi(s_\tau), \\
\text{variance } = \sigma_\phi(s_\tau), \\
\text{noise } = \epsilon$$

Then the value network consists of imagined value estimates $$V_R(s_\tau) = \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} r_n))$$ which is the sum of rewards until the end of a horizon, then using values $$v_\psi(s_\tau)$$ then computes $$V^{k}_N(s_\tau) = \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{h = \min(\tau+k, t+H)-1} \gamma^{n-\tau}r_n + \gamma^{h-\tau} v_\psi(s_h)))$$ as a estimate of rewards beyond $k$ steps with the learned value model, and $$V_\lambda(s_\tau)$$ which is a exponentially weighted average of $$V^{k}_N(s_\tau)$$ at different values of k, shown in the following. 

$$V_\lambda(s_\tau) = (1 - \lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V_{N}^{n}(s_\tau) + \lambda^{H-1}V_{N}^{H}(s_\tau)$$

This helps Dreamer do better with longer-term predictions of the world, over shortsightedness with other types of dynamics models for behavior learning. Since Dreamer disconnects the planning and action by training an actor and value network and uses analytic gradients and reparameterization, it is more efficient than PlaNet which searches the best actions among many predictions for different action sequences. This motivates the implementation of Dreamer to compare to PlaNet with potential performance improvements with a similar number of environment steps. The policy is trained via using the analytical gradient $$\nabla_\phi \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} V_\lambda(s_\tau))$$ from stochastic backpropagation, which in this case becomes a deterministic node where the action is returned shown in \ref{eq:1}, with the value network being updated with the gradient  $$\nabla_\psi \mathbb{E}(q_\theta q_\phi(\sum_{n=\tau}^{t+H} \frac{1}{2}\| v_\psi(s_\tau) - V_\lambda(s_\tau))\| ^2$$ after imagined value estimates are computed. All of this happens in the update steps for behavior and dynamics learning. Finally, in an environment interaction time step, the agent gets states from its history and returns actions from the action network, and value model estimates the imagined rewards that the action model gets in each state. These are trained cooperatively in a policy iteration fashion.

### Dreamer training

For training the first version of the Dreamer prototype, we used Cheetah the environment to compare directly to the in library PlaNet implementation. with action noise of $$\epsilon = 1.0 \text{ and } 0.3$$ like the original paper. The model, actor, and critic losses are logged from their respective networks. The model loss contains the reconstruction loss and represents the PlaNet world or dynamics model. The PlaNet world model is composed of an encoder and decoder, from a variational autoencoder (VAE) to transform the image inputs into latents, and then back to images. So, this encoder generates latents $$latents = encoder_\theta(img)$$ that get passed to the world model (RSSM) to get the posterior and prior with respect to latents and also an initial state and initial action shown as: $$s_0 = dynamics(img.shape), a_0$$. The RSSM returns the posterior and prior by rolling out an RNN from a starting state through each index of an embedding input, or latent, and an action such that we receive $$posterior, prior = RSSM_\theta(latents, s_0, a_0)$$ from the dynamics model. We then get features from the posterior $$features = dynamics(posterior)$$and use that for the image prediction, reward prediction

$$pred_{rew} = DenseRewNet_\theta(features) \\ pred_{img} = decoder_\theta(features)$$

 from the reward model defined by a dense network. We use these networks for losses from the image and reward, shown here (using probabilities from a softmax):
 
 $$loss_{img} = -\frac{1}{N}\sum \log(prob(pred_{rew} (observation)) \\
loss_{rew} = -\frac{1}{N}\sum \log(prob(pred_{img} (rew))$$

and pass the prior distribution and posterior distribution to a KL divergence loss:
  
 $$loss_{KL} = -\frac{1}{N}\sum KL(posterior \| prior)$$ 

and finally combine image loss, reward loss, and KL loss with a KL coeff to get the overall model loss. 

$$loss_{model} = loss_{rew} + loss_{img} + loss_{KL} * KLdiv_{const}$$


The actor loss is the loss from the actor network, which is described in the following:

$$continueprobability = \gamma * ones(rew) \\
\lambda-returns = (1 - \lambda)\sum_\lambda^{n-1}V_{N}^{n}(s_\tau) + \lambda^{H-1}V_{N}^{H}(s_\tau)$$


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


PlaNet learning how to turn (almost):

<iframe src="https://wandb.ai/mbrl_ducky/MBRL_Duckyt/reports/Shared-panel-22-04-28-19-04-83--VmlldzoxOTE2MDQz?highlightShare" style="border:none;height:700px;width:100%"> </iframe>

PlaNet learning to drive backward:

<iframe src="https://wandb.ai/mbrl_ducky/MBRL_Duckyt/reports/Shared-panel-22-04-28-19-04-48--VmlldzoxOTE2MDUw?highlightShare" style="border:none;height:1024px;width:100%"> </iframe>

Planet learning to jitter for speed reward:

<iframe src="https://wandb.ai/mbrl_ducky/MBRL_Duckyt/reports/Shared-panel-22-04-28-19-04-53--VmlldzoxOTE2MDYw?highlightShare" style="border:none;height:1024px;width:100%"> </iframe>


Dreamer learning to model Duckietown:


<iframe src="https://wandb.ai/mbrl_ducky/MBRL_Duckyt/reports/Shared-panel-22-04-28-19-04-57--VmlldzoxOTE1OTgz?highlightShare" style="border:none;height:500px;width:100%"> </iframe>

Dreamer trying to do ... something:

<iframe src="https://wandb.ai/mbrl_ducky/MBRL_Duckyt/reports/Shared-panel-22-04-28-19-04-06--VmlldzoxOTE2MDc3?highlightShare" style="border:none;height:1024px;width:100%"> </iframe>





<!-- <div>
<iframe src="https://wandb.ai/mbrl_ducky/MBRL_Duckyt/reports/Shared-panel-22-04-28-19-04-57--VmlldzoxOTE1OTgz?highlightShare" style="border:none;height:400px;width:100%"> </iframe>
</div> -->


## Conclusions

    8 [4 pts] Conclusions
    What have your results indicated?
    What have you learned? 
    What would you do differently next time? 
    Reflect on the scope of your project, was it too much? Why?


In this work we implemented the Dreamer algorithm for the MBRL-Lib library, and we applied MBRL approaches to the Gym-Duckietown environment. 

Our results indicate that there certainly is potential for model-based approaches to perform well in Gym-Duckietown, but there are certain barriers to overcome, mainly a better reward function is needed, and higher capacity models that plan much further into the future.

We learned a lot about how RL and MBRL algorithms are structured, how the distinction between model-based and model free can start to blur for algorithms like Dreamer. We learned how difficult it can be to design a reward function that enables learning to take place, and how long this process can take when each tiny modification requires hours of training to evaluate.

If we were to start over, we would ask for help earlier, lots of people know much more about Duckietown than we do and were very helpful, we imagine it is the same for MBRL-Lib, so we would have reached out to the authors earlier. 

Trying to fit in real robot experiments was definitely too much as both Gym-Duckietown and MBRL pose significant challenges to work with as it is. Finally, the amount of experiments needed to evaluate our work - since each RL run takes hours to learn on an environment like Duckietown - was out reach early on.



## Work Division

    Provide a description on what each group member is working on as part of the project. 
    I recommend each student work on most of the parts of the project so everyone learns about the content.
    Student Name: Did x, y, and z.
    Student Name: Did x, q, and r.
    Student Name: Did q, y, and r.














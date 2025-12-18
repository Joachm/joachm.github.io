---
layout: post
title: "Hypernetworks That Evolve Themselves"
date: 2025-12-18
---

Hypernetworks that Evolve Themselves
 
Hypernetworks are a class of deep neural networks that are tasked with producing the parameters for other neural networks. In this work, we let hypernetworks produce parameter updates to copies of themselves. Combined with a simple selection mechanism, this results in a new type of genetic algorithm, where the mutation mechanism is a heritable trait. In each generation, new hypernetworks find novel ways of producing parameters for policy networks and different ways of mutating the weights of their copies. In other words, a population of hypernetworks optimizes both their performance in a task and the optimization process itself.
 
**COPY VIDEO **
 ![Alt text](/assets/videos/hypernets/GHN_PP_video_mid.mp4)


 <video controls width="720" preload="metadata">
  <source src="/assets/videos/hypernets/GHN_PP_video_mod.mp4" type="video/mp4">
  Sorry — your browser doesn’t support embedded video.
</video>

As a starting point, we use Graph Hypernetworks (GHNs). GHNs have the desirable property that the same GHN can generate parameters for neural networks of different shapes and sizes. The way they do this is to treat the neural architecture of a target network as a graph to be processed by a graph neural network (GNN). In this graph, each node is a set of parameters of the target network, primarily the weights and biases of that network. The edges in the graph are how the signal from the input layer propagates through the network to the output layer. When the GHN is presented with the graph of a target network, each node in the network is represented by a learned vector embedding that depends on the type of the node (e.g., a fully connected weight matrix, a bias vector, etc). The graph neural network inside the GHN then updates each of these presentations through message passing along the edges of the graph. In this way, each node gets a unique representation that depends on the node’s placement in the architecture of the target network. Each of these vector representations is then used as input to the final module in the GHN: the hypernetwork. In the original GHN, the hypernetwork is simply a small multi-layer perceptron (MLP), the output dimension of which is as large as the largest number of parameters (P_max) in any of the sets of parameters in the target neural network. It is this output that is used for parameter generation in the target network. Each vector representation is parsed through this same MLP, resulting in P_max output values. Parameter sets that consist of P < P_max parameters are simply given the first P output values of the hypernetwork as parameters.
 
** GHN VIDEO
 
With this method, the study that originally introduced the GHN showed that GHNs can be trained to provide parameters for a range of different target networks, and even generalize to unseen target network architectures at test time. Our focus is different. If these GHNs can generate parameters for a set of different target networks, can they themselves be included in this set? By answering this question, we introduce Self-Referential Graph Hypernetworks: Hypernetworks that Evolve themselves.
Evolutionary Algorithms (EAs) have a long history as optimization algorithms for neural networks. They belong to the class of gradient-free optimization methods and thus provide a lot of flexibility in terms of the type of problems they can be applied to. Further, for reinforcement learning environments, EAs have been shown to have several advantages over gradient-based reinforcement learning algorithms, namely being better at handling delayed and sparse rewards, and having better chances of escaping local optima points.
While the family of EAs is large, any EA will have some generic traits in some form or another. The figures below show that self-referential GHNs mainly separate themselves from other EAs by localizing the mutation mechanism within the individuals, resulting in each individual having their own unique mutation mechanism.
 
** EVO ALGO PLOTS **
 
We change the original GHN framework in one important way: we have differentiated outputs for generating policy parameters and self-referential mutations. The main reasons to have a specialized output module for self-referential mutations are 1) to enable some stochasticity in the generation, such that the same parent can create non-identical offspring; and 2) mutations are additive instead of generating in-place values, as is done for the policy networks. The GNN remains the same for both cases. We make mutations randomly varied by using the node representations that have been updated by the GNN to parameterize a multi-dimensional Gaussian distribution: each representation is used as input to an MLP, and the result of this is used as the standard deviations for the distribution. We then draw a sample from this distribution and use it as input to the final MLP-layer of the hypernetwork that produces the weight updates for the specific node. Further, for each node in the graph of the architecture, we also calculate a node-specific learning rate, such that each part of the network can evolve at their own pace.
*** FULL SR-GHN OVERVIEW***
The selection mechanism stays simple: the individuals that have the highest fitness scores when evaluating them in a task get to stay in the population and reproduce. In other words, the only thing we directly select for is GHNs that produce high-performing policy networks. However, a subset of the self-referential GHN parameters are only used for generating mutations, so how are these optimized when we only select for policy performance? The explanation is actually straightforward. When a GHN is discovered in the population, we have actually found two things. First, we have found a GHN that is better than the others at generating weights for a good policy network. Further, the parent of this superior performer is likely to be better than the other parents in its ability to create offspring. And since the offspring also inherits its mutation parameters (with some variation) from its parent, the offspring is itself likely to be a better parent than most other parents. Thus, by simply selecting for the immediate performance by the policy network, we also indirectly select for a lineage that is likely to produce superior optimizers. This double effect is only possible because the mutation mechanism is itself a heritable trait and subject to mutations.
With this understanding in mind of how the self-referential GHNs can implement an evolution algorithm, let’s see it in action.
 
*** 2D TASK GRID; DIFFERENT SIZES ***
The figures above show the performance of the self-referential GHNs on classic 2-dimensional optimization benchmarks. One interesting aspect of this method is that we can completely decouple the number of parameters within the GHN from the number of dimensions in the solution we are trying to optimize. The sizes of the embedding vectors, as well as the other representations with the GHN, are all hyperparameters that can be set independently of the task dimension. The grid of figures shows how the self-referential GHNs perform differently on the same tasks with varying sizes of the inner representations of the GHN. For each column, a different hidden size was used. For simplicity, we use the same hidden size throughout the full GHN, even though all the intermediate representations could have different sizes. We see that the evolution of self-referential GHNs does not fare too well when the hidden sizes are just 2 or 4. However, when we increase the hidden size, the algorithm quickly converges on all the benchmarks and covers more of the optimal points in the 2D landscape. Importantly, this is without increasing the population or generation budget of the algorithm.
Since what makes the self-referential GHNs special as an evolutionary algorithm is that the population also evolves, how it changes over time, let’s test the algorithm in a non-stationary environment. Specifically, in the same 2D benchmarks, what happens if we shift the optimal points halfway through the evolution? Will the algorithm get stuck, or can it adapt to the new situation? Once again, we also vary the hidden size of the GHN representation.
*** 2D CHANGE ***
As we can see, when the hidden sizes are not too small, the population of self-referential GHNs easily adapts to changes. The advantages of a larger hidden size of the GHN representations can be explained by the possibility of changing more different aspects with a higher degree of independence than when the full representation of the mutations is compressed to a few values.
A hypothesis about self-referential GHNs is that the approach lets a population adapt more flexibly than other, more standard EAs. As explained above, the mutation mechanism itself will evolve over time to find higher-performing individuals, but that only explains part of the story as to why a population of GHNs can easily increase their exploration after the full population has converged to optimal points. The fact that we calculate a per-node learning rate means that when a population has found the optimal point in a loss landscape and does not have to adapt any more, the learning rates can go to very small values. This is actually likely to happen: if an individual has found a truly optimal area, then its most successful offspring will be ones that are close to it, i.e., offspring that were created with a small learning rate. However, when a learning rate is at a small value, the rest of the parameters of the mutation mechanism become less important; their consequences are multiplied by a small value anyway. This means that the evolutionary pressure on these values is relaxed, and they can drift to different values without affecting the fitness, resulting in a larger genetic variation within an otherwise converged population. If the loss landscape changes at some point in the future, the population can resume a wide exploration rapidly, because each individual can have mutation parameters that have drifted in a unique direction. All that needs to happen is that learning rates start to increase, and the population can quickly realize its genetic diversity.
With this in mind, let’s see how the self-referential GHNs compare to other evolution algorithms when it comes to recovering performance after a shift in the loss landscape. To differentiate between methods more clearly, we move to a slightly harder task. We use the classic control environment CartPole: this is the standard CartPole until generation 600, then the action meanings are flipped; at generation 1000, they flip back. This represents a clean and cheap way to force a real behavioral change without changing observations.
 
*** CartPole Training Curves***
*** Boxplots***
Even in this relatively simple environment, only our Self-Referential GHNs consistently recovered to optimal performance after a switch in the output order.
The other evolutionary algorithms fail in different ways. In the
videos below, we visualize the best-performing weights and their score for each
generation for different evolutionary approaches. As a qualitative
demonstration of the ability to switch between exploration and exploitation
phases, we see that self-referential GHNs simultaneously have the most rapid
adaptation after a switch and have the most stable parameters when an optimal solution has been found.

***Best Weight Evolution Videos**
 
From these videos, we can see that the Genetic Algorithm (GA) and OpenES (OES) fail due to insufficient change in their solutions after the change, whereas the Covariance Matrix-Adaptation Evolution Strategy (CMAES) becomes unstable even before the switch of action meanings occurs.
In the videos below, conduct the same experiment, but make the switch occur at a much earlier generation.
 
***Best Weight Evolution Videos - 300**
 
Here, GA and OES suffer from the same problems as before, but CMAES is actually able to adapt to the changing environment. The self-referential GHNs also perform well for this switch.
For good measure, we also try a much longer timescale:
 
***Best Weight Evolution Videos - 3000**
 
Again, only the self-referential GHNs perform well in this case.
 
Finally, we also test the self-referential GHN approach on a slightly harder task, the LunarLander. We once again switch the ordering of the actions at some point during the evolution, such that the population is forced to adapt.
 
***LunarLander Training Curves**
 
When we look at how the diversity of the population develops over time, we can see very clear spikes in diversity right after a switch in the environment has occurred. This, once again, confirms the ability of the GHNs to use their evolved mutation mechanism to automatically move from exploitation to exploration when needed.
 
Placing the mechanism of mutation-generation as learnable modules within individuals comes with some downsides. Namely, generating parameter updates is a much more computationally costly process than simply drawing the mutations from a distribution. Further, the size of the deterministic hypernetwork is determined by the maximum parameter requirement of any computational node in the target policy network that we want to generate parameters for. In turn, the size of the stochastic network is determined by the maximum parameter requirement of any computational node within the rest of the GHN, most often the output layer of the deterministic hypernetwork. This means the size of the Self-Referential GHN grows quickly with the size of the target networks, making the process of parameter generation even slower.
Future work should address this, for example, by replacing the updatable output layer of the deterministic hypernetwork with a random basis, just as we did with the output layer of the stochastic hypernetwork in this paper.
 
Another avenue of future research will be self-referential evolution with multiple target networks, instead of just a single one as was done in the experiments above. The Self-Referential GHNs maintain the potential of the original GHNs for generating parameters for neural networks with differing architectures, even ones not seen during the training phase.
 
Exploring this also opens the door for neural architecture search with mutations being applied to the neural architectures of the target networks as well as their parameters. In the extreme case, the Self-Referential GHNs could even mutate their own neural network architectures, creating even larger potentials for evolving high rates of diversity within the population. A possible direction is to combine the GHNs with a Neural Developmental Program such that development and evolution can interact.   
 
By demonstrating that purely neural systems can house the mutation machinery internally, this work takes a step toward being a closer analogy to natural evolution. We hope that this work will inspire the development of further research into novel evolution algorithms and neural networks that can improve themselves.


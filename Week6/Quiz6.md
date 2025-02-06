# Quiz 6

## How is unsupervised learning different from supervised learning? What is the goal of neural network for each learning type?

Supervised learning: models are trained on labeled data and are trained the mapping from input data to output data. The goal is to minimize the cost function and to make predictions on unseen data.

Unsupervised learning: models are trained on unlabeled data, no known categories and pre-defined labels. The models have to learn the data pattern on their own. The goal is to find the pattern and structure within the data.

## What are some examples of explicit and implicit density estimations in deep learning? For each method briefly describe their architecture.

Explicit density: Variational Autoencoders provide tractable density functions.

Inplicit density: Generative Adversarial Networks estimate density indirectly by generating samples or learning relative energy.

## How is Generative Adversarial Network different from FBVN and Variational autoencoder in inferring the density function?

GAN is implicit density model. It learns to generate samples that resemble the data distribution without estimating a likelihood function. The density is inferred implicitly through the competition between the generator and discriminator, where the generator produces samples and the discriminator evaluates their realism.

## What are the components of GAN? Briefly describe the role of each component.

Generator and Discriminator

Generator creates synthetic data samples that are as close as possible to the real data distribution. Its goal is to fool the discriminator into classifying its generated samples as "real."

Discriminator distinguishes between real data samples (from the true dataset) and fake samples (from the generator). Its goal is to correctly classify real samples as real and generated samples as fake.

### 1. **Generator**

- **Role**: The generator’s role is to create synthetic data samples that are as close as possible to the real data distribution. It learns to map a random noise vector (sampled from a simple distribution, like Gaussian or uniform) to the target data distribution.
- **Objective**: The generator’s goal is to **fool the discriminator** into classifying its generated samples as "real." This is achieved by learning to produce increasingly realistic samples over time.
- **Training Strategy**: The generator is trained by maximizing the probability that the discriminator misclassifies its output as real, effectively minimizing the discriminator's ability to distinguish between real and fake samples.

### 2. **Discriminator**

- **Role**: The discriminator acts as a binary classifier, distinguishing between real data samples (from the true dataset) and fake samples (from the generator).
- **Objective**: The discriminator’s goal is to **correctly classify real samples as real and generated samples as fake**. It provides feedback to the generator on how realistic the generated samples are.
- **Training Strategy**: The discriminator is trained to maximize its classification accuracy on both real and fake samples, effectively improving its ability to discern real data from generated data.

### How They Work Together

GANs are trained through an adversarial, or zero-sum, game between these two components:

- The **generator** tries to produce realistic samples to "fool" the discriminator.
- The **discriminator** tries to improve its accuracy in distinguishing real data from generated data.

## Briefly describe the optimization scheme used for training GAN. How does GAN optimize two competing cost functions?

GANs use a minimax optimization scheme, where the generator and discriminator are trained simultaneously but with opposing goals: the generator minimizes the probability that the discriminator correctly identifies fake samples, while the discriminator maximizes this probability. This results in two competing cost functions that are optimized iteratively; the generator improves by learning from the discriminator's feedback, and the discriminator refines itself based on new, more realistic samples from the generator.
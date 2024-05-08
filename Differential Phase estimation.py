import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tfd = tfp.distributions

# Generate synthetic data
np.random.seed(0)
num_samples = 100
d_true = np.pi / 4  # True differential phase
def generate_data(c_samples, noise):
    if noise == True:
        c_1_A = np.random.normal(0, np.pi/100, num_samples)
        c_2_A = np.random.normal(0, np.pi/100, num_samples)
        c_3_A = np.random.normal(0, np.pi/100, num_samples)
        c_1_B = np.random.normal(0, np.pi/100, num_samples)
        c_2_B = np.random.normal(0, np.pi/100, num_samples)
    else:
        c_1_A = 0
        c_2_A = 0
        c_3_A = 0
        c_1_B = 0
        c_2_B = 0
        
    y_A = c_1_A + (1 + c_2_A) * np.sin(d_true + c_samples + c_3_A)
    y_B = c_1_B + (1 + c_2_B) * np.sin(c_samples)
    return y_A, y_B
# Define the model
def model():
    c = yield tfd.Normal(loc=0., scale=1., name="c")

# Compile the model
@tf.function(autograph=False)
def joint_log_prob(num_samples, c):
    model_distribution = tfd.JointDistributionCoroutine(model)
    return model_distribution.log_prob(c)

# Define the target log probability function
def target_log_prob_fn(c):
    return joint_log_prob(num_samples, c)

# Run Hamiltonian Monte Carlo
num_results = 100
num_burnin_steps = 100

# Initial state for the chain
initial_state = [tf.ones(num_samples, dtype=tf.float32)*np.pi]

# Run HMC
hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.1,
    num_leapfrog_steps=3
)
hmc_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_state,
    kernel=hmc_kernel,
    trace_fn=lambda _, pkr: pkr.is_accepted

)

def StateFcn(noise):
    y_A, y_B = generate_data(c_sample.numpy(), noise)
    return y_A, y_B


all_y_A = []
all_y_B = []
all_y_A_true = []
all_y_B_true = []
plt.figure(figsize=(10, 8))
for c_sample in hmc_results[0]:
    y_A, y_B = StateFcn(True)
    y_A_true, y_B_true = StateFcn(False)
    all_y_A.extend(y_A)
    all_y_B.extend(y_B)
    all_y_A_true.extend(y_A_true)
    all_y_B_true.extend(y_B_true)

plt.plot(all_y_A, all_y_B, 'o', color='green', markersize=3, alpha=0.1, label = "Noisy")
plt.plot(all_y_A_true, all_y_B_true, color='red', markersize=3, alpha=0.1, label = "Noiseless")
plt.xlabel('y_A')
plt.ylabel('y_B')
plt.legend()
plt.title('Parametric Plot of y_B vs y_A for All Runs')
plt.grid(True)
plt.show()

# Plot posterior distribution (Need to still calculate the diff phase prob)
"""plt.figure(figsize=(8, 6))
plt.hist(np.array(hmc_results[0]).flatten(), bins=50, density=True, color='skyblue', edgecolor='black')
plt.xlabel('Differential Phase Estimate')
plt.ylabel('Posterior Density')
plt.title('Posterior Distribution of Differential Phase Estimates')
plt.grid(True)
plt.show()"""
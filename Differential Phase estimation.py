import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

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
num_results = 1000
num_burnin_steps = 1000

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


for c_sample in hmc_results[0]:
    y_A, y_B = StateFcn(True)
    y_A_true, y_B_true = StateFcn(False)
    all_y_A.extend(y_A)
    all_y_B.extend(y_B)
    all_y_A_true.extend(y_A_true)
    all_y_B_true.extend(y_B_true)
plt.figure(figsize=(10, 8))
plt.plot(all_y_A, all_y_B, 'o', color='green', markersize=3, alpha=0.1)
plt.plot(all_y_A_true, all_y_B_true, color='red', markersize=3, alpha=0.1)
plt.xlabel('y_A')
plt.ylabel('y_B')
plt.title('Parametric Plot of y_B vs y_A for All Runs')
plt.legend("With error", "without error")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(c_sample, y_A, 'o', color='green', markersize=3, alpha=0.1)
plt.plot(c_sample, y_B, 'o', color='green', markersize=3, alpha=0.1)
plt.xlabel('c_samples')
plt.ylabel('y')
plt.title('Parametric Plot of y_B vs y_A for All Runs')
plt.legend("With error", "without error")
plt.grid(True)
plt.show()

def likelihoodProb(y_A, y_B, c_sample, phi_d_est):
    equation = []
    c_sample = c_sample.numpy()
    for i in range(0,len(c_sample)-1):
        partA = np.exp(-(y_A[i] - np.sin(c_sample[i] + phi_d_est))*1/(2*(np.pi/50)))*np.exp(-(y_B[i] - np.sin(c_sample[i]+2*phi_d_est)))*1/(2*np.pi/50) + np.exp(-(y_B[i] - np.sin(c_sample[i]))*1/(2*np.pi/50))
        equation.append(partA)
    
    return equation
    
likelihoodNonInt = []
for i in range(0,num_results-1):
    likelihoodNonInt.append(likelihoodProb(y_A[i], y_B[i], c_sample[i], d_true))


x = np.arange(-np.pi/2,np.pi/2,np.pi/len(likelihoodNonInt[0]))
plt.figure(figsize=(10, 8))
for i in range(0,num_samples-1):
    plt.plot(x, likelihoodNonInt[i], 'o', color='green', markersize=3, label = "Noisy")
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
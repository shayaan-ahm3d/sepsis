import matplotlib.pyplot as plt
import numpy as np

# Delay in hours (0 to 6 hours)
hours = np.arange(0, 7, 1)

# Mortality increase per hour
# Starting from a baseline survival rate of 100%, we reduce survival each hour
initial_survival = 100
mortality_increase_rate = 0.076  # 7.6% per hour

# Compute survival rate after each hour of delay
survival_rates = initial_survival * (1 - mortality_increase_rate) ** hours
mortality_rates = 100 - survival_rates

# Plotting the chart
plt.figure(figsize=(8, 5))
plt.plot(hours, mortality_rates, marker='o', linestyle='-', linewidth=2)
plt.title("Increase in Mortality with Delayed Sepsis Treatment\n(Lee, J et al., 2017)")
plt.xlabel("Delay in Treatment (Hours)")
plt.ylabel("Cumulative Mortality (%)")
plt.grid(True)
plt.xticks(hours)
plt.ylim(0, 60)

plt.tight_layout()
plt.show()

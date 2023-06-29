import matplotlib.pyplot as plt
import time
import numpy as np

def plot_data():
    # Enable interactive mode
    plt.ion()
    
    plt.clf()

    # Create some data
    x = np.random.randn(10)
    y = np.random.randn(10)

    # Plot the data
    plt.plot(x, y,'o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Non-blocking Plot')
    plt.show(block=False)
    plt.pause(0.1)


# Continue executing the rest of your program
for i in range(100):
    print(i)
    time.sleep(1)  # Pause for 1 second
    if i % 10 == 0:
        plot_data()
    # Additional computations or operations can be performed here

# When you're ready to close the plot, call plt.close()
plt.close()
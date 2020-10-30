# Willis Hoke
# Display script for visualizing accuracy

import matplotlib.pyplot as plt 

# Parse file into two lists
trainAccs = []
testAccs = []
with open("results.txt") as f:
  for line in f:
    if line:
      accs = line.split()
      trainAccs.append(float(accs[0]))
      testAccs.append(float(accs[1]))

# Use pyplot to display values
def plotAccuracy(trainAccs, testAccs):
  epochs = len(testAccs)
  plt.plot(range(epochs), trainAccs)
  plt.plot(range(epochs), testAccs)
  plt.title("Accuracy over " + str(epochs-1) + " epochs")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy per epoch")
  plt.show()

plotAccuracy(trainAccs, testAccs)

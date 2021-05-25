import numpy as np
import cupy as cp
import random
import matplotlib.pyplot as plt

# builds a graph of the nodes of the network
class ForwardGraph:

    def __init__(self, size, types, l_rate, upper=0.9, lower=-0.9, l_rate_start=5000,momentum=True, mom_weight=0.333):
        self.size = size
        self.log_rate = 10
        self.iter = 0
        self.layer_functions = types
        self.l_rate = l_rate
        self.l_rate_start = l_rate_start
        self.weights = []
        self.bias = []
        self.node_values = []
        self.fdash_values = []
        self.momentum_weights = []
        self.momentum_bias = []
        self.deltas = []
        self.momentum = momentum
        self.mom_weight = mom_weight
        self.acc = []

        """ Creates matrices of the weights and bias.
        organises them such that we can multiply all the weights and inputs 
        pointing to the same node in the same matrix operation. """

        for i in range(len(size) - 1):
            self.weights.append(np.random.uniform(lower, upper, (size[i+1], size[i])))
            self.bias.append(np.random.uniform(upper, lower, (size[i+1], size[i])))
            self.momentum_weights.append(np.zeros([size[i+1], size[i]]))
            self.momentum_bias.append(np.zeros([size[i + 1], size[i]]))

        for i in range(len(size)):
            self.node_values.append(np.zeros(size[i]))
            self.fdash_values.append(np.zeros(size[i]))
            self.deltas.append(np.zeros(size[i]))

    def loop(self, inputs):
        self.node_values[0] = inputs[0]
        try:
            for i, j, n in zip(self.weights, self.bias, range(len(self.size))):
                for k, l, it in zip(i, j, range(len(i))):
                    # takes the weighted sum of the previous layer and adds the bias
                    weighted = np.multiply(self.node_values[n], k)
                    weighted_bias = np.add(weighted, l)
                    sum_n = np.sum(weighted_bias)
                    new_nodes = np.nan_to_num(self.layer_functions[n].compute(sum_n))
                    new_fDash = np.nan_to_num(self.layer_functions[n].fDash(sum_n))
                    self.node_values[n + 1][it] = new_nodes
                    self.fdash_values[n + 1][it] = new_fDash
         

        except:
            pass

        # calculating the 0 fDash values outside the main loop
        for i, j, n in zip(self.weights[0], self.bias[0], range(self.size[0])):
            weighted_f = np.multiply(self.node_values[0], i[n])
            weighted_bias_f = np.add(weighted_f, j[n])
            self.fdash_values[0][n] = self.layer_functions[0].fDash(np.sum(weighted_bias_f))

    def back_prop_sos(self, inputs):
        # calculates the last set of weights 1st
        self.deltas[-1] = self.node_values[-1] - inputs[1]
        self.acc.append(0.5*((np.sum(self.deltas[-1]))**2))

        for i, j, k, l in zip(self.weights[-1], self.fdash_values[-1], self.deltas[-1], self.bias[-1]):
            index = np.where(self.weights[-1] == i)
            temp = np.multiply(k, j)
            dEdW = temp*np.sum(self.node_values[-2])
            self.weights[-1][index] = np.add(i, -1*self.l_rate*self.l_rate_delta()*dEdW)
            self.bias[-1][index] = np.add(l, -1*self.l_rate*self.l_rate_delta() * temp)

            if self.momentum and self.iter != 0:
                self.weights[-1][index] = np.add(self.weights[-1][index], (self.momentum_weights[-1][index]*self.l_rate*self.mom_weight))
                self.bias[-1][index] = np.add(self.bias[-1][index], (self.momentum_bias[-1][index]*self.l_rate*self.mom_weight))

            if self.momentum:
                self.momentum_weights[-1][index] = -1 * self.l_rate*self.l_rate_delta() * dEdW
                self.momentum_bias[-1][index] = -1 * self.l_rate*self.l_rate_delta()* temp



        # update the remaining weights
        for k, n in zip(reversed(self.fdash_values[:-1]), reversed(range(1, len(self.size)-1))):
            
            """
            creates a matrix for each layer of the delta vector 
            multiplied by the  node values of that layer
            the subtracts this matric from the matrix of 
            weights
            """
      
            # calculating the deltas for a given layer
            deltas = self.deltas[n+1]
            weights = self.weights[n]
            temp1 = np.transpose(np.multiply(deltas, np.transpose(weights)))


            def multiply_row_by_deltas(row):
                return np.multiply(row, k)

            temp2 = np.apply_along_axis(multiply_row_by_deltas, 1, temp1)
            new_deltas = np.sum(temp2, axis=0)
            new_deltas = np.nan_to_num(new_deltas)
            self.deltas[n] = new_deltas

            # updating the weights
            delta_weights = np.outer(new_deltas, self.node_values[n-1])
            temp3 = -1*self.l_rate*self.l_rate_delta()*delta_weights
            self.weights[n-1] = np.add(temp3, self.weights[n-1])
  


            # updating the bias'
            delta_bias = np.ones(np.shape(self.bias[n-1]))
            for delt, iter in zip(new_deltas, range(len(new_deltas))):
                delta_bias[iter] = np.multiply(delta_bias[iter], delt)
            delta_bias = -1*self.l_rate*self.l_rate_delta()*delta_bias
            self.bias[n-1] = np.add(self.bias[n-1], delta_bias)

            if self.momentum and self.iter != 0:
                self.weights[n-1] = np.add(self.weights[n-1], self.momentum_weights[n-1] * self.l_rate*self.l_rate_delta()*self.mom_weight)
                self.bias[n-1] = np.add(self.bias[n-1], self.momentum_bias[n-1] * self.l_rate*self.l_rate_delta()*self.mom_weight)

            if self.momentum:
                self.momentum_weights[n - 1] = temp3
                self.momentum_bias[n-1] = delta_bias

        # updating the learning rate
        self.iter += 1

    def predict(self, input):
        self.loop(input)
        return self.node_values[-1]

    def l_rate_delta(self):
        # decreasing the learning rate over time
        if self.iter > self.l_rate_start:
            self.log_rate += 0.001
            return 1/np.log10(self.log_rate)
        else:
            return 1
         
       
    def print_acc(self):
        x = [i for i in range(len(self.acc))]
        plt.plot(x, self.acc)
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.show()

    def print_nodes(self):
        print(self.node_values)

    def print_weights(self):
        print(self.weights)
        
    def print_deltas(self):
        print(self.deltas)

    def acc(self):
        return self.acc


class Sigmoid:
    def __init__(self):
        pass

    def compute(self, z):
        return 1 / (1 + np.exp(-z))

    def fDash(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

class Tanh:
    def __init__(self):
        pass

    def compute(self, z):
        return np.tanh(z)

    def fDash(self, z):
        return ((np.cosh(z)**2) - (np.sinh(z)**2))/(np.cosh(z)**2)

class Relu:
    def __init__(self):
        pass

    def compute(self, z):
        return np.maximum(z, 0.0)

    def fDash(self, z):
        if np.maximum(z, 0) == 0:
            return 0.0
        else:
            return 1.0


class Ensemble:

    def __init__(self, number_of_graphs, structure, upper, lower, l_rate, l_rate_start ,momentum, mom_weight):

        self.list_of_graphs = []
        for i in range(number_of_graphs):
            self.list_of_graphs.append(ForwardGraph(structure[i][0], structure[i][1], l_rate[i], upper[i], lower[i], l_rate_star[i], momentum[i], mom_weight[i]))

    def loop(self, data):
        for graph in self.list_of_graphs:
            graph.loop(data)

    def learn(self, data):
        for graph in self.list_of_graphs:
            graph.back_prop_sos(data)

    def predict(self, input):
        predict = self.list_of_graphs[0].predict(input)
        for i in self.list_of_graphs[1:]:
            predict = np.add(predict, i.predict(input))
        predict = predict*(1/len(self.list_of_graphs))
        return predict

    def average_acc(self):
        avg_acc = self.list_of_graphs[0].acc
        for i in self.list_of_graphs[1:]:
            avg_acc = np.add(avg_acc, i.acc)
        avg_acc = avg_acc*(1/len(self.list_of_graphs))
        return avg_acc

    def print_avg_acc(self):
        x = [i for i in range(len(self.average_acc()))]
        plt.plot(x, self.average_acc())
        plt.xlabel("Iterations")
        plt.ylabel("Average Error")
        plt.show() 
        
    def graphs(self):
        return self.list_of_graphs

                                           
                                                            
                                                            
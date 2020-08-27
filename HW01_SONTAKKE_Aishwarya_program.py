__author__ = 'AISHWARYA SONTAKKE'

"""
Author: AISHWARYA SONTAKKE as4897

This program completes tasks 1 to 5 assigned in HW1 for CSCI 720.
1. Exploratory Data Analysis
2. 1D Clustering using Otsu's method
3. Graphing
4. Exploring Regularization
5. Using quantization based on height
"""

import math
import pandas
import numpy as np
import warnings
import matplotlib.pyplot as plt

def quantization(raw_data, bin_size):
    '''
    This method takes the raw data and quantizes it based on the passed
    bin_size.
    :param raw_data: Data to b quantized
    :param bin_size: size of the interval
    :return: quantized_data
    '''
    quantized_data = []

    #Looping over all elements of raw data and quantizing on the given
    #formula
    for element in raw_data:
        quantized_data.append(float(math.floor(float(element) / bin_size)
                                    * bin_size))

    #Sorting quantized data
    quantized_data.sort()
    return quantized_data

def otsu_one_dimension(data, reg_flag):
    '''
    This method implements otsu method for 1D clustering/binarizing data
    :param data: Given data to be binarized
    :param reg_flag: Determines whether regularization should be implemented
                     0 = no regularization, 1 = with regularization
    :return:best_threshold : best threshold obtained for splitting
            best_mixed_variance : minimum mixed variance determining
            mixed_variances_list :list of all mixed variances
            best_alpha : 99999 for no regularization else
                        alpha that gives the best split for binarizing data
    '''

    #Initializing variables
    best_mixed_variance = float('inf')
    best_threshold = float('inf')
    best_cost = float('inf')
    best_alpha = 99999
    mixed_variance_list =[]
    best_cost_list = []

    #Considering every point in data as threshold and splitting according to
    #that threshold
    for threshold in data:
        weight_left = []
        weight_right = []

        # Add data points <= threshold to weight_left list and rest to
        # weight_right list
        for current_point in data:
            if float(current_point) <= float(threshold):
                weight_left.append(float(current_point))
            elif float(current_point) > float(threshold):
                weight_right.append(float(current_point))

        #Getting average of the weight lists to further calculate mixed
        #variance
        weight_left_fraction = float(len(weight_left))/float(len(data))
        weight_right_fraction = float(len(weight_right))/float(len(data))

        #Calculating variance using numpy libraries
        variance_left = np.var(weight_left)
        variance_right = np.var(weight_right)

        #If results are to be found with regularization
        if reg_flag == 1:

            #Initializing given variables
            NormFactor = 100
            given_list = [100, 1, 1 / 5, 1 / 10, 1 / 20, 1/22, 1/24, 1 / 25,
                          1/27, 1/30, 1/35, 1/40, 1/45, 1 / 50, 1/55, 1/60,
                          1/61, 1/70, 1/74, 1/79, 1/85, 1/92, 1 / 100,
                          1/800, 1/950, 1 / 1000]

            #Calculating mixed variance(in this case objective function)
            mixed_variance = weight_left_fraction * variance_left + \
                             weight_right_fraction * variance_right

            #Looping through every value of alpha to find best regularization
            #and consecutively best cost fucntion
            for alpha in given_list:

                #Formula based on the given question
                regularization = float(abs(len(weight_left) -
                                    len(weight_right)) / NormFactor) * alpha

                #Determining cost fuction
                cost_function = mixed_variance + regularization

                #Finding best threshold value based on lowest cost function
                if cost_function < best_cost:
                    best_cost = cost_function
                    best_alpha = alpha
                    best_threshold_reg = threshold

            #Adding best costs to the list
            best_cost_list.append(best_cost)
        else:

            # Calculating mixed variance so no regularization
            mixed_variance = weight_left_fraction * variance_left + \
                             weight_right_fraction * variance_right

            # Adding mixed variances to the list
            mixed_variance_list.append(mixed_variance)

            # Finding best threshold value based on lowest mixed variation
            if mixed_variance < best_mixed_variance:
                best_mixed_variance = mixed_variance
                best_threshold = threshold
    '''
    #Generating clusters
    cluster_left = []
    cluster_right = []
    for current_point in data:
        if float(current_point) <= float(best_threshold):
            cluster_left.append(float(current_point))
        elif float(current_point) > float(best_threshold):
            cluster_right.append(float(current_point))
    '''

    if reg_flag == 1:
        return best_threshold_reg, best_cost, \
               best_cost_list, best_alpha
    else:
        return best_threshold, best_mixed_variance, \
               mixed_variance_list, best_alpha

def main():
    '''
    The main method.
    :return:
    '''

    print ("Part 1: Exploratory Data Analysis")
    mystery_data_frame = pandas.read_csv("Mystery_Data_2195.csv")

    #Taking the first column from mystery_data_frame to calculate average
    # and standard deviation
    column_one = mystery_data_frame.columns[0]
    average = mystery_data_frame[column_one].mean()
    standard_deviation = mystery_data_frame[column_one].std()
    print("Average : ", average)
    print("Standard deviation : ", standard_deviation)

    #Removing last value from the data
    new_mystery_data_frame = mystery_data_frame[:len(mystery_data_frame.
                                                     index)-1]
    new_average = new_mystery_data_frame[column_one].mean()
    print ("Removing last value from the data")
    print ("Updated average : ", new_average)

    ############################################################################
    print ("\nPart 2: 1D Clustering using Otsu\'s method on the age")

    #Using pandas to read csv file
    snowfolks_data_frame = pandas.read_csv\
        ("Abominable_Data_For_Clustering__v44.csv")

    #Converting pandas dataframe series object to a list
    snowfolks_data_age = snowfolks_data_frame[snowfolks_data_frame.columns[0]]\
        .values.tolist()

    #Quantizing snowfolks age into bins of 2 years interval
    quantized_age = quantization(snowfolks_data_age, 2)

    #Using filterwarnings to not print warnings due to computation of variance
    # in otsu_method
    warnings.filterwarnings("ignore")

    #Calling otsu method without regularization to find best split
    best_age, age_best_mixed_variance, age_mixed_variances_list, age_alpha \
        = otsu_one_dimension(quantized_age, 0)

    print("Best age to separate two clusters: ", best_age)
    print("Minimum mixed variance: ", age_best_mixed_variance)

    ############################################################################
    print("\nPart 3: Graphing")
    print("Plot for Mixed variance for snowfolk’s data based on age "
          "vs quantized age is generated and saved as Plot.png")

    plt.plot(age_mixed_variances_list, quantized_age)
    plt.title('Mixed variance for snowfolk’s data based on age vs '
              'quantized age')
    plt.xlabel('Mixed variances')
    plt.ylabel('Quantized age')
    plt.plot(age_best_mixed_variance,best_age, marker='o', color = 'red')
    plt.show()
    plt.savefig('Plot')

    ############################################################################
    print("\nPart 4: Exploring Regularization")
    best_age_reg, age_best_mixed_variance_reg, age_mixed_variances_list_reg, \
        age_alpha_reg = otsu_one_dimension(quantized_age, 1)
    print("Best age to separate two clusters (with regularization) : ",
          best_age_reg)
    print("Minimum mixed variance (with regularization): ",
          age_best_mixed_variance_reg)
    print("Best value of alpha for the split :", age_alpha_reg)

    ############################################################################
    print ("\nPart 5: Use quantization based on height.")

    #Converting pandas dataframe series object to a list
    snowfolks_data_height = snowfolks_data_frame[snowfolks_data_frame.
        columns[1]].values.tolist()

    #Performing quantization with bin_size 5
    quantized_height = quantization(snowfolks_data_height, 5)

    # Calling otsu method without regularization to find best split
    best_height, best_mixed_variance_height, height_mixed_variance_list, \
        height_alpha = otsu_one_dimension(quantized_height, 0)
    print("Best height : ", best_height)
    print("Minimum mixed variance for height : ", best_mixed_variance_height)

if __name__ == '__main__':
    main()
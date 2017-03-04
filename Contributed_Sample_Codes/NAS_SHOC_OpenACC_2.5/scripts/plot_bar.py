#!/usr/bin/env python
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

import sys
import math
import collections

from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    '''
    Need an argument with the data file where each row has the format

    "'benchmark': CLASS: EXEC_TIME"

    e.g.

    plot_boxplot.py file.data abs "x label" "y label" A B C D

    'BT' : A: 0.1
    'BT' : A: 0.2

    this will accumulate the values for BT_th1 = abs(0.1-0.2) = 0.1

    Comments you can use C++-style inline comments //

    '''
    args = sys.argv

    USAGE = '''USAGE: python plot_bar.py <data file> op xlabel ylabel Class [Class..].
            opt = operation to perform with the two matching values. "+, -, abs, /, 1/"
            xlabel = label to show in the x axis
            ylabel = label to show in the y axis
            data file = has to have a pair of rows for each benchmark
            e.g. python plot_bar.py file.data abs x y A B C D, will compute abs(X1-X2)'''

    if len(args) < 6:
        print USAGE
        sys.exit(-1)

    dataFile = args[1]
    op = args[2]
    x_label = args[3]
    y_label = args[4]
    classes = args[5:]

    if op not in ["+","-","/","abs", "1/"]:
        print USAGE
        sys.exit(-1)

    print "Processing dataFile: " + dataFile + ", classes: " + str(classes)

    benchmarks = collections.OrderedDict()
    benchmarks_array = list()

    # reading the data file to build the dict
    data = open(dataFile, "r")

    line = data.readline().strip()
    while len(line) > 0:
        # remove comments
        line = line[0:line.find("//")]
        split_line = line.split(":")

        b_name = split_line[0].strip('\'')
        b_class = split_line[1].strip()
        tmp = split_line[2].strip()
        b_exectime = float(tmp if len(tmp) > 0 else -1)

        if b_class in classes:

            if b_name not in benchmarks:
                # creating benchmark info
                # dict for benchmark data
                benchmarks_array.append(b_name)

                tmp_dict = dict()
                for key in classes:
                    tmp_dict[key] = [None, None]
                benchmarks[b_name] = tmp_dict

            benchmark_info = benchmarks[b_name]

            # check if first value
            if benchmark_info[b_class][0] == None:
                benchmark_info[b_class][0] = b_exectime if b_exectime >= 0 else None
            else:
                benchmark_info[b_class][1] = b_exectime

        line = data.readline().strip()

    # number of columns in the plot
    N = len(benchmarks)
    print "Benchmarks considered: " + str(benchmarks.keys())
    print "Data matrix [number benchmarks, number of classes] = [" + str(N) + ", " + str(len(classes)) + "]"
    print str(benchmarks)

    # data
    data = np.zeros((N, len(classes)))

    for bnc_pos in range(len(benchmarks_array)):
        for class_pos in range(len(classes)):
            b_name = benchmarks_array[bnc_pos]
            b_class = classes[class_pos]

            # executing the operation
            x1, x2 = benchmarks[b_name][b_class]

            res = 0
            if x1 != None and x2 != None:
                # perform op
                if op == "+":
                    res = x1 + x2
                elif op == "-":
                    res = x1 - x2
                elif op == "/":
                    res = x1 / x2
                elif op == "1/":
                    res = x2 / x1
                elif op == "abs":
                    res = abs(x1 - x2)
                else:
                    print "ERROR: op not defined!: " + op
                    sys.exit(-1)

            data[bnc_pos][class_pos] = res

    # saving processed data
    print str(data)
    proFile = open(dataFile + ".op", "w")
    proFile.write("Benchmarks considered: " + str(benchmarks.keys()) + "\n")
    proFile.write("Classes: " + str(classes) + "\n")
    proFile.write("op: " + op + "\n")
    proFile.write("Data matrix [number benchmarks, number of classes] = [" + str(N) + ", " + str(len(classes)) + "]\n")
    proFile.write(str(benchmarks) + "\n")
    proFile.write(str(data))
    proFile.close()

    num_columns = len(classes)
    plt.grid(True, lw=0.5, c='.5')

    ind = np.arange(num_columns)  # the x locations for the groups
    width = 0.1

    #colors = ['b', 'g', 'r', 'c', 'k', 'y', 'm']
    hatches = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo', 'OOO', '...', '***']

    plus_width = 0
    axs = []
    for bnc_pos in range(len(benchmarks_array)):
        b_name = benchmarks_array[bnc_pos]
        # checking range
        end = 0
        for ith in range(len(classes)):
            if data[bnc_pos][ith] > 0:
                end+=1
            else:
                break
        x = ind + plus_width
        #axs.append(plt.bar(x, data[bnc_pos], width, color=colors[bnc_pos]))
        axs.append(plt.bar(x, data[bnc_pos], width, fill=True, color='w', edgecolor='k', hatch=hatches[bnc_pos]))
        plus_width += width

    print "Max: " + str(data.max())
    maxY = data.max()
    indY = np.arange(start=0, stop=maxY+0.5, step=0.5)
    indYn = [""] * len(indY)
    for i in range(int(math.ceil(maxY))):
        indYn[2*i] = str(i)

    plt.xticks(ind + plus_width/2, classes)
    plt.yticks(indY, indYn)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.minorticks_on()
    #plt.legend(axs, benchmarks_array, loc='center right', framealpha=0.5)
    plt.legend(axs, benchmarks_array, loc='upper left', framealpha=0.5)
    fig = plt.figure(1)

    plt.show()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.relim()
    ax.autoscale()

    try:
        # Initialize:
        pp = PdfPages(dataFile + ".pdf")
        pp.savefig(fig)
        plt.close()
        pp.close()

    except IOError, e:
        print e

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

    plot_indbar.py file.data n "x label" "y label" A B C D

    'BT' : A: 0.1
    'CG' : A: 0.2

    Comments you can use C++-style inline comments //

    '''
    args = sys.argv

    USAGE = '''USAGE: python plot_indbar.py <data file> op xlabel ylabel Class [Class..].
            opt = operation to perform. "n (normal), l2 (log2), l10 (log10)"
            xlabel = label to show in the x axis
            ylabel = label to show in the y axis
            data file = has to have a pair of rows for each benchmark
            e.g. python plot_indbar.py file.data l x y A B C D, will show the values in log scale'''

    if len(args) < 6:
        print USAGE
        sys.exit(-1)

    dataFile = args[1]
    op = args[2]
    x_label = args[3]
    y_label = args[4]
    classes = args[5:]

    if op not in ["n", "l2", "l10"]:
        print USAGE
        sys.exit(-1)

    print "Processing dataFile: " + dataFile + ", classes: " + str(classes)

    benchmarks = collections.OrderedDict()
    benchmarks_array = list()
    runtimes_array = list()

    # reading the data file to build the dict
    data = open(dataFile, "r")

    line = data.readline().strip()
    while len(line) > 0:
        # remove comments
        line = line[0:line.find("//")]
        split_line = line.split(":")

        tmp = split_line[0].strip('\'')
        tmp_split = tmp.split(";")
        b_name = tmp_split[0]
        b_runtime = tmp_split[1] if len(tmp_split) == 2 else None

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
                    tmp_dict[key] = dict()
                benchmarks[b_name] = tmp_dict

            benchmark_info = benchmarks[b_name]

            if b_runtime not in runtimes_array:
                runtimes_array.append(b_runtime)

            if b_runtime != None:
                benchmark_info[b_class][b_runtime] = b_exectime

        line = data.readline().strip()

    # number of columns in the plot
    N = len(benchmarks)
    print "Benchmarks considered: " + str(benchmarks.keys())
    #print "Data matrix [number of benchmarks, number of classes] = [" + str(N) + ", " + str(len(classes)) + "]"
    dataInfo = "Data matrix [# classes, # runtimes, # benchmarks] = [" + str(len(classes)) + ", " + str(len(runtimes_array)) + ", " + str(N) + "]"
    print dataInfo
    print str(benchmarks)

    # data
    data = np.zeros((len(classes), len(runtimes_array), N))

    for bnc_pos in range(len(benchmarks_array)):
        for class_pos in range(len(classes)):
            b_name = benchmarks_array[bnc_pos]
            b_class = classes[class_pos]

            for run_pos in range(len(runtimes_array)):
                b_runtime = runtimes_array[run_pos]
                # executing the operation
                x1 = benchmarks[b_name][b_class][b_runtime]

                res = 0
                if x1 != None:
                    # perform op
                    if op == "n":
                        res = x1
                    elif op == "l2":
                        res = math.log(x1)/math.log(2)
                    elif op == "l10":
                        res = math.log(x1)/math.log(10)
                    else:
                        print "ERROR: op not defined!: " + op
                        sys.exit(-1)

                data[class_pos][run_pos][bnc_pos] = res

    # saving processed data
    print str(data)

    proFile = open(dataFile + ".op", "w")
    proFile.write("Benchmarks considered: " + str(benchmarks.keys()) + "\n")
    proFile.write("Classes: " + str(classes) + "\n")
    proFile.write("op: " + op + "\n")
    proFile.write(str(benchmarks) + "\n")
    proFile.write(dataInfo + "\n")
    proFile.write(str(data))
    proFile.close()

    num_columns = N #len(classes)
    plt.grid(True, lw=0.5, c='.5')

    ind = np.arange(num_columns)  # the x locations for the groups
    width = 0.1

    #colors = ['b', 'g', 'r', 'c', 'k', 'y', 'm', 'tan', 'darkorange']
    hatches = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo', 'OOO', '...', '***']

    plus_width = 0
    axs = []
    class_pos = 0
    for run_pos in range(len(runtimes_array)):

        x = ind + plus_width
        axs.append(plt.bar(x, data[class_pos][run_pos], width, fill=True, color='w', edgecolor='k', hatch=hatches[run_pos]))
        #axs.append(plt.bar(x, data[bnc_pos], width, color=colors[bnc_pos]))
        plus_width += width

    print "Max: " + str(data.max())
    if op != "n":
        maxY = data.max()
        indY = np.arange(start=0, stop=maxY+0.5, step=0.5)
        indYn = [""] * len(indY)
        for i in range(int(math.ceil(maxY))):
            indYn[2*i] = str(i)
        plt.yticks(indY, indYn)

    plt.xticks(ind + plus_width/2, benchmarks_array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.minorticks_on()
    plt.legend(axs, runtimes_array, loc='upper right', framealpha=1.0)
    #plt.legend(axs, benchmarks_array, loc=2, framealpha=0.5, borderaxespad=0., bbox_to_anchor=(1.05, 1))
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

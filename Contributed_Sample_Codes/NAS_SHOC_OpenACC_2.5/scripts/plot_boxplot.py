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

    "'benchmark;num_threads': CLASS: EXEC_TIME"

    e.g.

    plot_boxplot.py file.data abs "x label" "y label" A B C D

    'BT;1' : A: 0.1
    'BT;1' : A: 0.2

    this will accumulate the values for BT_th1 = abs(0.1-0.2) = 0.1

    Comments you can use C++-style inline comments //

    '''
    args = sys.argv

    USAGE = '''USAGE: python plot_boxplot.py <data file> op xlabel ylabel Class [Class..].
            opt = operation to perform with the two matching values. "+, -, abs, /, 1/"
            xlabel = label to show in the x axis
            ylabel = label to show in the y axis
            data file = has to have a pair of rows for each benchmark;num_threads
            e.g. python plot_boxplot.py file.data abs x y A B C D, will compute abs(X1-X2)'''

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
    threads_array = list()
    benchmarks_array = list()

    # reading the data file to build the dict
    data = open(dataFile, "r")

    line = data.readline().strip()
    while len(line) > 0:
        # remove comments
        line = line[0:line.find("//")]
        split_line = line.split(":")

        [b_name, b_nthreads] = split_line[0].strip('\'').split(";")
        b_class = split_line[1].strip()
        tmp = split_line[2].strip()
        b_exectime = float(tmp if len(tmp) > 0 else 0)
        if b_exectime == 0.0:
            b_exectime = sys.float_info.min

        if b_class in classes:
            if b_nthreads not in threads_array:
                threads_array.append(b_nthreads)

            if b_name not in benchmarks:
                # creating benchmark info
                # dict for benchmark data
                benchmarks_array.append(b_name)
                benchmarks[b_name] = collections.OrderedDict()

            benchmark_info = benchmarks[b_name]
            if b_nthreads not in benchmark_info:
                tmp_dict = dict()
                for key in classes:
                    tmp_dict[key] = [None, None]
                benchmark_info[b_nthreads] = tmp_dict

            # check if first value
            if benchmark_info[b_nthreads][b_class][0] == None:
                benchmark_info[b_nthreads][b_class][0] = b_exectime if b_exectime >= 0 else None
            else:
                benchmark_info[b_nthreads][b_class][1] = b_exectime

        line = data.readline().strip()

    # number of columns in the plot
    N = len(benchmarks)
    print "Benchmarks considered: " + str(benchmarks.keys())
    print "thread array: " + str(threads_array)
    print "Data matrix [number of classes, number benchmarks, number of threads_array] = [" + str(len(classes)) + ", " + str(N) + ", " + str(len(threads_array)) + "]"
    print str(benchmarks)

    # data
    data = dict()
    for cla in classes:
        data[cla] = np.zeros((N, len(threads_array)))

    for bnc_pos in range(len(benchmarks_array)):
        for th_pos in range(len(threads_array)):
            for b_class in classes:
                b_name = benchmarks_array[bnc_pos]
                b_nthreads = threads_array[th_pos]

                # executing the operation
                x1, x2 = benchmarks[b_name][b_nthreads][b_class]

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

                data[b_class][bnc_pos][th_pos] = res

    # saving processed data
    print str(data)
    proFile = open(dataFile + ".op", "w")
    proFile.write("Benchmarks considered: " + str(benchmarks.keys()) + "\n")
    proFile.write("Classes: " + str(classes) + "\n")
    proFile.write("op: " + op + "\n")
    proFile.write("thread array: " + str(threads_array) + "\n")
    proFile.write("Data matrix [number of classes, number benchmarks, number of threads_array] = [" + str(len(classes)) + ", " + str(N) + ", " + str(len(threads_array)) + "]\n")
    proFile.write(str(benchmarks) + "\n")
    proFile.write(str(data))
    proFile.close()

    # demonstrate how to toggle the display of different elements:
    num_columns = len(classes)

    medianprops = dict(linewidth=2.5, color='back')
    meanpointprops = dict(markeredgecolor='black',
                      markerfacecolor='black')

    # labels
    xticks = threads_array
    if len(classes) == 1:
        cl = classes[0]
        plt.title("Class " + cl)
        plt.grid(True, lw=0.5, c='.5')
        b = plt.boxplot(data[cl], labels=xticks, showmeans=True, meanline=False, medianprops=medianprops, meanprops=meanpointprops)

        for name, line_list in b.iteritems():
            for line in line_list:
                line.set_color('k')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.minorticks_on()
        fig = plt.figure(1)
    else:
        fig, axes = plt.subplots(1, ncols=num_columns, sharey=(op == "/" or op == "1/"))

        for i in range(len(classes)):
            cl = classes[i]
            axes[i].set_title("Class " + cl)
            axes[i].grid(True, lw=0.5, c='.5')
            b = axes[i].boxplot(data[cl], labels=xticks, showmeans=True, meanline=False, medianprops=medianprops, meanprops=meanpointprops)

            for name, line_list in b.iteritems():
                for line in line_list:
                    line.set_color('k')

            axes[i].set_xlabel(x_label)
            axes[i].minorticks_on()

        axes[0].set_ylabel(y_label)

    if op != "/" or op != "1/":
        fig.subplots_adjust(wspace=0.4)

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

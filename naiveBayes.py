#!/usr/bin/env python2

# Authors: Mariah Martin Shein, Kassy Raymond and Andrew Hamilton-Wright
#
# Description:
#   Naive Bayes Classifier
#
#   Formats data by dividing by class, assuming has already been
#   discretized
#
#   Then trains classifier
#
#   Mariah 2013
#    - adapted from code by Krishnamurthy Viswanathan

import sys
import math
import string
from StringIO import StringIO
from numpy import genfromtxt
import numpy

##
class NaiveBayes:
    'Naive Bayes Classifier'

    def __init__(self):
        self.counts = {} #keys are tuples (className,featureName,featureValue)
                    #and values are number of times these occur together
        self.featureNames = []
        self.classNames = []    # holds the names of each class
        self.classCounts = []    # holds the number of instances of each class
        self.numClasses = 0
        self.total = 0.0        # holds the total number of instances

    ##
    def trainBayes(self, ifile, verbose=0):
        # read in the actual data from the file
        success = 0

        allData = numpy.genfromtxt(ifile, dtype='str',  delimiter=',', skip_header=1)

        try:
            infile = open(ifile, "r")
        except IOError, (errno, strerror):
            sys.stderr.write("Failure opening input file '%s'\n" % ifile)
            sys.stderr.write("    I/O error(%s) : %s\n" % (errno, strerror))
            sys.stderr.write("\n")
            printHelp()
            sys.exit(1)

        line = infile.readline()
        if len(line) == 0:
            sys.stderr.write("No data in file\n")
            sys.exit(1)

        spaceToken = line.split()
        commaToken = spaceToken[0].split(',')
        try:
            value = float(commaToken[0])
            sys.stderr.write("Error: file does not contain class names!\n")
            sys.exit(1)
        except ValueError:
            #split line into tokens and add to self.featureNames
            for tok in spaceToken:
                tokenName = tok.split(',')
                self.featureNames.append(tokenName[0])

        # count number of instances of each feature value and add to self.counts
        for row in allData:
            #print row
            self.total += 1
            endRow = len(row)-1
            cName = row[endRow]
            #print "This is the name: " +  str(cName)
            if cName not in self.classNames:
                self.classNames.append(cName)
                self.numClasses += 1
            tempCount = 0
            while cName != self.classNames[tempCount]:
                tempCount += 1
            if len(self.classCounts) == len(self.classNames):
                self.classCounts[tempCount] += 1
            else:
                self.classCounts.append(1)
            featNum = 0
            #for each row, check class, feature name and value
            while featNum < endRow:
                featName = self.featureNames[featNum]
                #featVal = int(row[featNum])
                featVal = row[featNum].strip()
                # if this tuple has been added to self.counts, increment the value by 1
                if (cName, featName, featVal) in self.counts:
                    self.counts[(cName, featName, featVal)] += 1
                # else, add to self.counts
                else:
                    #give base value of 2 instead of 1 to compensate for
                    #smoothing that happens in classify function
                    self.counts[(cName, featName, featVal)] = 2
                featNum += 1
        infile.close()

    ##
    def testBayes(self, testFilename, outFilename, verbose=0):
        valVector = []

        #read in a file and classify each line by calling classify
        try:
            infile = open(testFilename, "r")
        except IOError, (errno, strerror):
            sys.stderr.write("Failure opening input file '%s'\n" % testFilename)
            sys.stderr.write("    I/O error(%s) : %s\n" % (errno, strerror))
            sys.stderr.write("\n")
            printHelp(argv[0])
            sys.exit(1)

        line = infile.readline()
        if len(line) == 0:
            sys.stderr.write("No data in file\n")
            success = -1
            return success

        #print the classification to outFilename if given

        if outFilename == None:
            outfile = sys.stdout
        else:
            try:
                outfile = open(outFilename, "w")
            except IOError, (errno, strerror):
                sys.stderr.write("Failure opening input file '%s'\n" % outFilename)
                sys.stderr.write("    I/O error(%s) : %s\n" % (errno, strerror))
                sys.stderr.write("\n")
                printHelp(argv[0])
                sys.exit(1)

        line = infile.readline()
        if len(line) == 0:
            sys.stderr.write("No data in file\n")
            success = -1
            return success

        spaceToken = line.split()
        commaToken = spaceToken[0].split(',')
        try:
            value = float(commaToken[0])
        except ValueError:
            line = infile.readline()
            spaceToken = line.split()

        # loop through file and pass each line to classify
        while len(line) != 0:
            valVector = []
            for tok in spaceToken:
                newTok = tok.split(',')
                valVector.append(newTok[0])
            name = self.classify(valVector)
            outfile.write(name)
            outfile.write("\n")
            line = infile.readline()
            spaceToken = line.split()
        outfile.close()
        infile.close()

    ##
    def classify(self, featLine):
        #calculates the probability of the given data point for each possible class,
        # then returns the most likely
        maxProb = 0
        maxClass = "No class"
        for cn in self.classNames:
            classTot = self.classCounts[self.classNames.index(cn)]
            classProb = classTot/self.total
            for feat in featLine:
                logProb = 0
                featName = self.featureNames[featLine.index(feat)]
                #f = int(feat)
                f = feat.strip()
                if (cn, featName, f) in self.counts:
                    logProb += math.log(float(self.counts[(cn, featName, f)])/classTot)
                else:
                    # if (class, name, value) tuple does not exist, assume value of 1
                    logProb += math.log(1.0/classTot)
            classProb *= math.exp(logProb)
            if classProb > maxProb:
                maxProb = classProb
                maxClass = cn
        return maxClass

##
def printHelp(progname):

    sepIndex = string.rfind(progname, "/");
    if sepIndex < 0:
        printProgname = progname
    else:
        printProgname = progname[sepIndex + 1:]
    print printProgname, "[ options ] <TRAIN-DATA> <TEST-DATA> [<OUTPUT-NAME>]"
    print ""
    print "Options:"
    print "    -v : make more verbose"
    print "    -h : print help"
    print ""
    print "Format data and train Naive Bayes Classifier"



##
## --------------------------
## Beginning of mainline
##

def main(argv):
    verbosity = 0
    trainFile = None
    testFile = None
    outFile = None

    for arg in argv[1:]:
        if arg[0] == '-':
            if arg[1] == 'h' or arg[1] == 'H':
                printHelp(argv[0]);
                sys.exit(1)
            elif arg[1] == 'v':
                verbosity = verbosity + 1
            else:
                print "Unknown option '%s' - ignoring" % arg
        else:
            if trainFile == None:
                trainFile = arg
            elif testFile == None:
                testFile = arg
            elif outFile == None:
                outFile = arg
            else:
                print "Too many filename!"
                printHelp(argv[0]);
                sys.exit(1)

    if testFile == None:
        print "Need training and testing file names"
        printHelp(argv[0])

    bay = NaiveBayes()

    bay.trainBayes(trainFile, verbose=verbosity)
    bay.testBayes(testFile, outFile, verbose=verbosity)

    sys.exit(0);

##
## ----------------------------
##
##
if __name__ == "__main__": main(sys.argv)

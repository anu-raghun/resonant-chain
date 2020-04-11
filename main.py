import testruns as test

def main():
    '''boxsweepdata0 draws the box models for each start point and duration'''
    '''boxsweepdata1 sweeps through the data, and for each box width, calculates the diffLL values for all iterations of start points'''
    test.boxSweepData0()
    test.boxSweepData1()

main()



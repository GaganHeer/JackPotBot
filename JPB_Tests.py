import JPB_DQfD as dqfd
import numpy as np

def testANN(model, testInput):
    dqTest, nstepTest, slmcTest = model.predict([testInput, testInput, testInput])

    testInput = testInput[0].tolist()
    dqTest = dqTest[0].tolist()
    nstepTest = nstepTest[0].tolist()
    slmcTest = slmcTest[0].tolist()

    assert dqTest != testInput
    print("\n\n\nDQ input state different from DQ Output")
    assert nstepTest != testInput
    print("NStep input state different from NStep Output")
    assert slmcTest != testInput
    print("SLMC input state different from SLMC Output")
    print("------------------------------\nPassed ANN test\n")

def testOutput(model, testInput, outputSize):
    dqTest, nstepTest, slmcTest = model.predict([testInput, testInput, testInput])
    assert len(dqTest[0]) == outputSize
    print("DQ output is sized correctly")
    print("DQ output is categorized correctly")
    assert len(nstepTest[0]) == outputSize
    print("NStep output is sized correctly")
    print("NStep output is categorized correctly")
    assert len(slmcTest[0]) == outputSize
    print("SLMC output is sized correctly")
    print("SLMC output is categorized correctly")
    print("------------------------------\nPassed Output test\n")

def testNaN(model, testInput):
    dqTest, nstepTest, slmcTest = model.predict([testInput, testInput, testInput])

    testInput = testInput[0].tolist()
    dqTest = dqTest[0].tolist()
    nstepTest = nstepTest[0].tolist()
    slmcTest = slmcTest[0].tolist()
    nan = np.nan

    for val in dqTest:
        assert val != nan
    print("DQ QVals are not NaN")

    for val in nstepTest:
        assert val != nan
    print("NStep QVals are not NaN")

    for val in slmcTest:
        assert val != nan
    print("SLMC QVals are not NaN")
    print("------------------------------\nPassed NaN test\n")

def testInf(model, testInput):
    dqTest, nstepTest, slmcTest = model.predict([testInput, testInput, testInput])

    testInput = testInput[0].tolist()
    dqTest = dqTest[0].tolist()
    nstepTest = nstepTest[0].tolist()
    slmcTest = slmcTest[0].tolist()
    posInf = np.inf
    negInf = np.NINF

    for val in dqTest:
        assert val != posInf
        assert val != negInf
    print("DQ QVals are not Infinite")

    for val in nstepTest:
        assert val != posInf
        assert val != negInf
    print("NStep QVals are not Infinite")

    for val in slmcTest:
        assert val != posInf
        assert val != negInf
    print("SLMC QVals are not Infinite")
    print("------------------------------\nPassed Inf test\n")

def runTestBed(model, testInput, outputSize):
    testANN(model, testInput)
    testOutput(model, testInput, outputSize)
    testNaN(model, testInput)
    testInf(model, testInput)
    print("==============================\nALL TESTS PASSED\n==============================")
    
